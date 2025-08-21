# scripts/predict.py
import sys
import os
import pandas as pd
import numpy as np
import json
import joblib
import xgboost as xgb

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from f1_optimizer import config, utils, strategy_optimizer

# scripts/predict.py (함수 교체 제안)
def create_simulation_data(preprocessor, multi_model, pitstop_model, params_data, numerical_cols, categorical_cols):
    """
    학습된 모델과 파라미터를 기반으로 시뮬레이션용 랩 데이터를 생성합니다. (개선된 버전)
    """
    race_laps = params_data['metadata'].get('total_laps', 58)
    laps_range = range(1, race_laps + 1)
    
    # 1. 모델에 입력할 가상의 레이스 데이터를 '트랙별 중앙값'으로 생성
    
    # params.json에서 트랙별 피처 중앙값 불러오기
    feature_medians = params_data.get("feature_medians", {})
    
    base_X_data = {}
    all_cols = numerical_cols + categorical_cols

    for col in all_cols:
        # 저장된 중앙값이 있으면 사용하고, 없으면 0 또는 'MEDIUM'으로 대체
        default_value = 'MEDIUM' if col in categorical_cols else 0
        base_X_data[col] = [feature_medians.get(col, default_value)] * race_laps

    # 랩마다 변해야 하는 값들은 별도로 덮어쓰기
    base_X_data['LapNumber'] = laps_range
    base_X_data['TyreLife'] = laps_range # 스틴트 시작 시점부터의 타이어 수명

    # Stint는 가상으로 1-스톱 시나리오를 가정
    if 'Stint' in base_X_data:
        pit_lap_guess = int(race_laps / 2)
        base_X_data['Stint'] = [1] * pit_lap_guess + [2] * (race_laps - pit_lap_guess)


    X_sim = pd.DataFrame(base_X_data)[all_cols] # 컬럼 순서 고정
    
    # 2. 전처리
    X_sim_transformed = preprocessor.transform(X_sim)
    
    # 3. 모델로 예측 수행
    y_pred_multi = multi_model.predict(X_sim_transformed)
    y_pred_pitstop_proba = pitstop_model.predict_proba(X_sim_transformed)[:, 1]

    # 4. 시뮬레이션용 최종 DataFrame 생성
    lap_data_df = pd.DataFrame({
        'lap': X_sim['LapNumber'].values,
        'laptime': y_pred_multi[:, 0],
        'tyre_wear': y_pred_multi[:, 1],
        'pit_proba': y_pred_pitstop_proba,
        'rainfall': X_sim['Rainfall'].values if 'Rainfall' in X_sim.columns else np.zeros(len(X_sim)),
        'track_temp': X_sim['TrackTemp'].values if 'TrackTemp' in X_sim.columns else np.full(len(X_sim), 30.0)
    })
    
    return lap_data_df.groupby('lap').mean().reset_index().sort_values('lap')



def main(track_name, initial_compound):
    print(f"--- {track_name} 트랙({initial_compound} 시작) 전략 예측 시작 ---")
    
    # 1. 학습된 Artifacts 로드
    params_path = f"{config.PARAMS_JSON_PATH}/{track_name}_complete_strategy_params_v2.json"
    preprocessor_path = f"{config.ARTIFACTS_PATH}/models/{track_name}_preprocessor.joblib"
    multi_model_path = f"{config.ARTIFACTS_PATH}/models/{track_name}_multi_output_model.json"
    pitstop_model_path = f"{config.ARTIFACTS_PATH}/models/{track_name}_pitstop_classifier.json"

    try:
        params_data = utils.load_json(params_path)
        
        # --- 주석 해제 및 실제 파일 로딩 ---
        preprocessor = joblib.load(preprocessor_path)
        
        multi_model = xgb.XGBRegressor()
        multi_model.load_model(multi_model_path)
        
        pitstop_model = xgb.XGBClassifier()
        pitstop_model.load_model(pitstop_model_path)
        # ------------------------------------
        
        print("✅ 모든 모델과 파라미터를 성공적으로 불러왔습니다.")

    except FileNotFoundError as e:
        print(f"❌ 에러: 필요한 파일({e.filename})을 찾을 수 없습니다. 먼저 train.py를 실행하세요.")
        sys.exit(1)

    # 2. 시뮬레이션용 데이터 생성 (실제 모델 예측 사용)
    #    preprocessor로부터 컬럼 정보를 역으로 추출합니다.
    num_cols = preprocessor.transformers_[0][2]
    cat_cols = preprocessor.transformers_[1][2]
    
    print(">>> 시뮬레이션용 랩 데이터 생성 중...")
    lap_data_df = create_simulation_data(
        preprocessor, multi_model, pitstop_model, params_data, num_cols, cat_cols
    )
    print(f"✅ 시뮬레이션 데이터 생성 완료. (총 {len(lap_data_df)}랩)")

    # 3. 전략 최적화 실행
    race_laps = params_data['metadata'].get('total_laps', 58)
    final_strategy = strategy_optimizer.run_optimization(
        lap_data_df, params_data, initial_compound, race_laps
    )

    # 4. 결과 저장 및 출력
    result_filename = f"{config.RESULT_JSON_PATH}/{track_name}_{initial_compound}_strategy.json"
    utils.save_json(final_strategy, result_filename)
    
    print("\n--- 최종 최적 전략 ---")
    print(json.dumps(final_strategy, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("사용법: python scripts/predict.py [track_name] [initial_compound]")
        print("예시: python scripts/predict.py Monaco MEDIUM")
        sys.exit(1)
    
    track = sys.argv[1]
    compound = sys.argv[2]
    main(track, compound)