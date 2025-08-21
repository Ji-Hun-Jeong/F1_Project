# scripts/train.py
import sys
import os
from datetime import datetime
import joblib
import json

# 프로젝트 루트 경로를 sys.path에 추가
# 이 스크립트의 위치(scripts/)에서 두 단계 상위 폴더(f1_strategy_optimizer/)를 기준으로 경로를 잡아야 합니다.
# 현재 코드 구조상 PROJECT_ROOT_PATH를 직접 사용합니다.
sys.path.append("/home/azureuser/cloudfiles/code")
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)


from f1_optimizer import config, data_loader, preprocessing, model_trainer, parameter_calculator, utils

def run_training_pipeline():
    for track_name in config.TRACKS_TO_PROCESS:
        try:
            print(f"\n{'='*20} {track_name} 트랙 학습 파이프라인 시작 {'='*20}")
            
            # 1. 데이터 로드
            laps_df, weather_df, race_laps = data_loader.load_and_merge_data(track_name)
            
            # 2. 데이터 전처리
            laps_df = preprocessing.fix_time_sorting(laps_df)
            if not weather_df.empty:
                weather_df = preprocessing.fix_time_sorting(weather_df)
            
            merged_df = preprocessing.merge_weather_data(laps_df, weather_df)

            # [핵심 수정] Stint 정보를 가장 먼저 계산
            df_with_stints = preprocessing.calculate_stints(merged_df)

            processed_df = preprocessing.feature_engineer(df_with_stints)
            
            # 3. 데이터셋 분리
            laps_data_grouped = processed_df.copy()
            laps_data_single = processed_df[processed_df['GrandPrix'].str.contains(track_name, na=False)].copy()
            dry_laps_data_grouped = laps_data_grouped[laps_data_grouped.get('Rainfall', 0) == 0].copy()
            wet_laps_data_grouped = laps_data_grouped[laps_data_grouped.get('Rainfall', 0) > 0].copy()
            dry_laps_data_single = laps_data_single[laps_data_single.get('Rainfall', 0) == 0].copy()
            has_wet_data = not wet_laps_data_grouped.empty
            
            # 4. 모델 학습
            preprocessor, num_cols, cat_cols = preprocessing.get_preprocessor(processed_df)
            multi_model, pitstop_model, best_params, X_test, X_test_transformed = model_trainer.train_all_models(
                processed_df, preprocessor, num_cols, cat_cols
            )
            
            # 5. 파라미터 계산
            X_dry_test_grouped = preprocessor.transform(dry_laps_data_grouped[num_cols + cat_cols])
            y_dry_pitstop_test_grouped = (dry_laps_data_grouped['PitInTime'].notna()).astype(int)
            y_dry_pitstop_proba_grouped = pitstop_model.predict_proba(X_dry_test_grouped)[:, 1]
            
            dry_pit_proba_threshold = parameter_calculator.calculate_pit_proba_threshold(y_dry_pitstop_test_grouped, y_dry_pitstop_proba_grouped)
            dry_tyrelife_threshold = parameter_calculator.calculate_tyre_wear_threshold(dry_laps_data_grouped)
            dry_compound_performance = parameter_calculator.calculate_all_compound_performance(dry_laps_data_grouped)
            dry_compound_choice_thresholds = parameter_calculator.find_compound_choice_thresholds(dry_laps_data_single, track_name)
            dry_pit_stop_time = parameter_calculator.calculate_pit_stop_time(dry_laps_data_single, track_name)
            
            feature_medians = {}
            cols_to_calc_median = [col for col in num_cols if col in laps_data_single.columns]
            for col in cols_to_calc_median:
                feature_medians[col] = laps_data_single[col].median()


            # 6. 결과물 JSON 통합
            final_json_data = {
                "metadata": {
                    "track": track_name,
                    "version": "2.0",
                    "total_laps": race_laps,
                    "created_at": datetime.now().isoformat()
                },
                "feature_medians": feature_medians,
                "dry_strategy_params": {
                    "ml_hyper_parameters": best_params,
                    "pit_proba_threshold": dry_pit_proba_threshold,
                    "TyreLife_threshold": dry_tyrelife_threshold,
                    "compound_choice_thresholds": dry_compound_choice_thresholds,
                    "pit_stop_time": dry_pit_stop_time,
                    "compound_performance": {k: v for k, v in dry_compound_performance.items() if k in ['SOFT', 'MEDIUM', 'HARD']}
                }
            }
            if has_wet_data:
                wet_compound_performance = parameter_calculator.calculate_all_compound_performance(wet_laps_data_grouped)
                wet_pit_stop_time = parameter_calculator.calculate_pit_stop_time(wet_laps_data_grouped, track_name)
                final_json_data["wet_strategy_params"] = {
                    "pit_stop_time": wet_pit_stop_time,
                    "compound_performance": {k: v for k, v in wet_compound_performance.items() if k in ['INTERMEDIATE', 'WET']}
                }

            # 7. Artifacts 저장
            params_filename = f"{config.PARAMS_JSON_PATH}/{track_name}_complete_strategy_params_v2.json"
            utils.save_json(final_json_data, params_filename)

            preprocessor_filename = f"{config.ARTIFACTS_PATH}/models/{track_name}_preprocessor.joblib"
            os.makedirs(os.path.dirname(preprocessor_filename), exist_ok=True)
            joblib.dump(preprocessor, preprocessor_filename)
            print(f"✅ Preprocessor가 '{preprocessor_filename}'으로 저장되었습니다.")

            multi_model_filename = f"{config.ARTIFACTS_PATH}/models/{track_name}_multi_output_model.json"
            multi_model.save_model(multi_model_filename)
            print(f"✅ Multi-Output 모델이 '{multi_model_filename}'으로 저장되었습니다.")
            
            pitstop_model_filename = f"{config.ARTIFACTS_PATH}/models/{track_name}_pitstop_classifier.json"
            pitstop_model.save_model(pitstop_model_filename)
            print(f"✅ PitStop 분류 모델이 '{pitstop_model_filename}'으로 저장되었습니다.")

        except Exception as e:
            print(f"!!!!!! ERROR processing {track_name}: {e} !!!!!!")
            continue # 다음 트랙으로 넘어감

if __name__ == "__main__":
    run_training_pipeline()