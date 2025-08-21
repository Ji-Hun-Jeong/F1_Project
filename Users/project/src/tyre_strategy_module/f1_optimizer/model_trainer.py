# f1_optimizer/model_trainer.py
import xgboost as xgb
import optuna
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import ADASYN
from sklearn.metrics import r2_score, precision_score, f1_score, roc_curve

def train_all_models(laps_data, preprocessor, numerical_cols, categorical_cols):
    train_data = laps_data[laps_data['Rainfall'] == 0].copy()

    X = train_data[numerical_cols + categorical_cols]
    y_laptime = train_data['LapTime'].values
    y_pitstop = (train_data['PitInTime'].notna()).astype(int).values
    y_tyrewear = train_data['TyreWear'].values
    y_multi = np.column_stack((y_laptime, y_pitstop, y_tyrewear))

    X_train, X_test, y_multi_train, y_multi_test = train_test_split(
        X, y_multi, test_size=0.2, random_state=42
    )
    
    preprocessor.fit(X_train)
    X_train_transformed = preprocessor.transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    y_laptime_train, y_pitstop_train, y_tyrewear_train = y_multi_train.T
    y_laptime_test, y_pitstop_test, y_tyrewear_test = y_multi_test.T

    adasyn = ADASYN(sampling_strategy=0.3, random_state=42)
    X_train_pit_res, y_pitstop_train_res = adasyn.fit_resample(X_train_transformed, y_pitstop_train)

    def objective(trial):
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 5, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
            'random_state': 42,
            'objective': 'reg:squarederror'
        }
        
        reg_param = param.copy()
        reg_param['objective'] = 'reg:squarederror'
        multi_output_model = xgb.XGBRegressor(**reg_param)
        multi_output_model.fit(X_train_transformed, y_multi_train[:, [0, 2]])

        clf_param = param.copy()
        clf_param['objective'] = 'binary:logistic'
        clf_param['scale_pos_weight'] = trial.suggest_float('scale_pos_weight', 3, 5)
        xgb_pitstop = xgb.XGBClassifier(**clf_param)
        xgb_pitstop.fit(X_train_pit_res, y_pitstop_train_res)

        y_pred_multi = multi_output_model.predict(X_test_transformed)
        y_pred_pitstop_proba = xgb_pitstop.predict_proba(X_test_transformed)[:, 1]

        y_pred_laptime = y_pred_multi[:, 0]
        y_pred_tyrewear = y_pred_multi[:, 1]
        
        fpr, tpr, thresholds = roc_curve(y_pitstop_test, y_pred_pitstop_proba)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx] if len(thresholds) > 0 else 0.5
        y_pred_pitstop = (y_pred_pitstop_proba >= optimal_threshold).astype(int)

        precision = precision_score(y_pitstop_test, y_pred_pitstop, zero_division=0)
        f1 = f1_score(y_pitstop_test, y_pred_pitstop, zero_division=0)
        r2_laptime = r2_score(y_laptime_test, y_pred_laptime)
        r2_tyre_wear = r2_score(y_tyrewear_test, y_pred_tyrewear)
        
        score = 0.4 * f1 + 0.2 * precision + 0.2 * r2_laptime + 0.2 * r2_tyre_wear
        return score

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30, n_jobs=-1)
    
    best_params = study.best_params
    
    multi_output_params = best_params.copy()
    multi_output_params.pop('scale_pos_weight', None)
    multi_output_params['objective'] = 'reg:squarederror'
    final_multi_output_model = xgb.XGBRegressor(**multi_output_params)
    final_multi_output_model.fit(X_train_transformed, y_multi_train[:, [0, 2]])

    pitstop_params = best_params.copy()
    pitstop_params['objective'] = 'binary:logistic'
    final_xgb_pitstop = xgb.XGBClassifier(**pitstop_params)
    final_xgb_pitstop.fit(X_train_pit_res, y_pitstop_train_res)

    return final_multi_output_model, final_xgb_pitstop, best_params, X_test, X_test_transformed