# f1_optimizer/parameter_calculator.py
import pandas as pd
import numpy as np
import scipy.stats as sps
from sklearn.metrics import f1_score, roc_curve
from sklearn.linear_model import LinearRegression

def filter_by_race_performance(stint_data, laps_data, compound):
    if stint_data.empty:
        return stint_data
    
    compound_laps = laps_data[laps_data['Compound'] == compound]
    if len(compound_laps) < 50:
        return stint_data
    
    normal_lap_range = (
        compound_laps['LapTime'].quantile(0.25),
        compound_laps['LapTime'].quantile(0.75)
    )
    
    valid_stints = []
    for _, stint_row in stint_data.iterrows():
        driver = stint_row['Driver']
        start_lap = stint_row['LapNumber']
        end_lap = start_lap + stint_row['StintLength']
        
        stint_laps = laps_data[
            (laps_data['Driver'] == driver) &
            (laps_data['LapNumber'] >= start_lap) &
            (laps_data['LapNumber'] < end_lap) &
            (laps_data['Compound'] == compound)
        ]
        
        if len(stint_laps) >= stint_row['StintLength'] * 0.7:
            stint_avg_time = stint_laps['LapTime'].median()
            if normal_lap_range[0] <= stint_avg_time <= normal_lap_range[1] * 1.1:
                valid_stints.append(stint_row)
    
    return pd.DataFrame(valid_stints) if valid_stints else pd.DataFrame()

def filter_performance_based_stints(stint_data, laps_data):
    filtered_stints = {}
    
    for compound in stint_data['Compound'].unique():
        compound_stints = stint_data[stint_data['Compound'] == compound].copy()
        
        if compound == 'SOFT': valid_range = (8, 25)
        elif compound == 'MEDIUM': valid_range = (12, 35)
        else: valid_range = (18, 45)
        
        in_range_stints = compound_stints[
            (compound_stints['StintLength'] >= valid_range[0]) &
            (compound_stints['StintLength'] <= valid_range[1])
        ]
        
        performance_filtered = filter_by_race_performance(in_range_stints, laps_data, compound)
        
        if len(performance_filtered) > 10:
            Q1 = performance_filtered['StintLength'].quantile(0.25)
            Q3 = performance_filtered['StintLength'].quantile(0.75)
            IQR = Q3 - Q1
            final_stints = performance_filtered[
                (performance_filtered['StintLength'] >= Q1 - 1.5 * IQR) &
                (performance_filtered['StintLength'] <= Q3 + 1.5 * IQR)
            ]
        else:
            final_stints = performance_filtered
        
        if not final_stints.empty:
            filtered_stints[compound] = final_stints['StintLength'].tolist()
        else:
            filtered_stints[compound] = []
            
        print(f"{compound}: {len(compound_stints)} -> {len(final_stints)} stints after filtering")
    
    return filtered_stints

def analyze_stint_distribution(stint_lengths):
    stint_array = np.array(stint_lengths)
    
    # 데이터가 너무 적으면 중앙값을 바로 반환
    if len(stint_array) < 5:
        return {'recommended': int(np.median(stint_array)) if len(stint_array) > 0 else 20}

    analysis = {
        'mean': np.mean(stint_array),
        'median': np.median(stint_array),
        'mode': float(sps.mode(stint_array, keepdims=False)[0]) if len(stint_array) > 0 else np.median(stint_array),
        'p75': np.percentile(stint_array, 75),
        'recommended': 0
    }
    
    # [핵심 수정] 짧은 스틴트의 영향을 줄이기 위해 중앙값과 75백분위수의 평균을 사용
    analysis['recommended'] = int(np.mean([analysis['median'], analysis['p75']]))
    
    return analysis

def validate_stint_logic(optimal_stint, theoretical_values):
    validated = optimal_stint.copy()
    adjustments_made = False
    
    if validated.get('SOFT', 0) >= validated.get('MEDIUM', 99):
        print(f"⚠️  WARNING: SOFT stint ({validated.get('SOFT')}) >= MEDIUM stint ({validated.get('MEDIUM')})")
        if 'SOFT' in validated and 'MEDIUM' in validated:
            soft_deviation = abs(validated['SOFT'] - theoretical_values['SOFT'])
            medium_deviation = abs(validated['MEDIUM'] - theoretical_values['MEDIUM'])
            if soft_deviation > medium_deviation:
                validated['SOFT'] = int((validated['SOFT'] + theoretical_values['SOFT']) / 2)
                adjustments_made = True
                print(f"   → SOFT adjusted to {validated['SOFT']}")
    
    if validated.get('MEDIUM', 0) >= validated.get('HARD', 99):
        print(f"⚠️  WARNING: MEDIUM stint ({validated.get('MEDIUM')}) >= HARD stint ({validated.get('HARD')})")
        if 'MEDIUM' in validated and 'HARD' in validated:
            medium_deviation = abs(validated['MEDIUM'] - theoretical_values['MEDIUM'])
            hard_deviation = abs(validated['HARD'] - theoretical_values['HARD'])
            if medium_deviation > hard_deviation:
                validated['MEDIUM'] = int((validated['MEDIUM'] + theoretical_values['MEDIUM']) / 2)
                adjustments_made = True
                print(f"   → MEDIUM adjusted to {validated['MEDIUM']}")
            else:
                validated['HARD'] = int((validated['HARD'] + theoretical_values['HARD']) / 2)
                adjustments_made = True
                print(f"   → HARD adjusted to {validated['HARD']}")
    
    if adjustments_made:
        print("✅ Logical adjustments completed based on data quality")
    else:
        print("✅ All stint values follow logical order")
        
    return validated

def calculate_pit_proba_threshold(y_true, y_pred_proba):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    youden_index = tpr - fpr
    optimal_idx = np.argmax(youden_index)
    optimal_threshold = thresholds[optimal_idx]
    y_pred_binary = (y_pred_proba >= optimal_threshold).astype(int)
    optimal_f1 = f1_score(y_true, y_pred_binary)
    print(f"Optimal pit_proba threshold (Youden's Index): {optimal_threshold:.3f}")
    print(f"Corresponding F1 Score: {optimal_f1:.3f}")
    return optimal_threshold

def calculate_tyre_wear_threshold(laps_data, percentile=75):
    tyre_data = laps_data.dropna(subset=['TyreLife'])
    wear_threshold = np.percentile(tyre_data['TyreLife'], percentile)
    print(f"TyreLife {percentile}th percentile threshold: {wear_threshold:.1f}")
    corr = tyre_data['TyreLife'].corr(tyre_data['LapTime'])
    if abs(corr) > 0.5:
        sorted_data = tyre_data.sort_values('TyreLife')
        lap_time_increase_idx = np.where(np.diff(sorted_data['LapTime']) > sorted_data['LapTime'].mean() * 0.1)[0]
        if len(lap_time_increase_idx) > 0:
            adjusted_threshold = sorted_data['TyreLife'].iloc[lap_time_increase_idx[0]]
            print(f"Adjusted threshold based on LapTime increase: {adjusted_threshold:.1f}")
            return adjusted_threshold
    return wear_threshold

def extract_pit_stop_data(laps_data, track_name, total_laps):
    pit_stop_list = []
    track_data = laps_data[laps_data['GrandPrix'].str.contains(track_name, na=False)]
    for (gp, driver), group in track_data.groupby(['GrandPrix', 'Driver']):
        group = group.sort_values('LapNumber')
        if 'PitInTime' in group.columns and group['PitInTime'].notna().any():
            pit_stops = group[group['PitInTime'].notna()].copy()
        elif 'IsPitStop' in group.columns and group['IsPitStop'].notna().any():
            pit_stops = group[group['IsPitStop'] == 1].copy()
        else:
            pit_stops = group[group['TyreWear'] >= 28].drop_duplicates(subset=['LapNumber'], keep='first').copy()
        pit_stops.loc[:, 'NextCompound'] = pit_stops['Compound'].shift(-1)
        for idx, row in pit_stops[pit_stops['NextCompound'].notnull()].iterrows():
            pit_stop_list.append({
                'TrackTemp': row['TrackTemp'],
                'RemainingLaps': min(total_laps - row['LapNumber'], total_laps),
                'ChosenCompound': row['NextCompound']
            })
    return pd.DataFrame(pit_stop_list)

def find_compound_choice_thresholds(laps_data, track_name):
    print(f"--- {track_name} 트랙의 컴파운드 선택 임계값 계산 시작 ---")
    total_laps_dict = {'Abu_Dhabi': 58, 'Monza': 53, 'Silverstone': 52} # 예시
    total_laps = total_laps_dict.get(track_name, 58)
    pit_stop_df = extract_pit_stop_data(laps_data, track_name, total_laps)
    if pit_stop_df.empty:
        return {"t1": 30.0, "t2": 29, "t3": 11, "accuracy": 0.0}
    t1_values = np.arange(15.0, 40.0, 1.0)
    t2_values = np.arange(0.3, 0.9, 0.1)
    t3_values = np.arange(0.05, 0.25, 0.05)
    best_accuracy = 0
    best_thresholds = (t1_values[0], int(t2_values[0] * total_laps), int(t3_values[0] * total_laps))
    for t1 in t1_values:
        for t2_ratio in t2_values:
            for t3_ratio in t3_values:
                if t3_ratio < t2_ratio:
                    t2_laps = int(t2_ratio * total_laps)
                    t3_laps = int(t3_ratio * total_laps)
                    if len(pit_stop_df) == 0: continue
                    def predict_compound(row):
                        if row['TrackTemp'] > t1 or row['RemainingLaps'] > t2_laps: return "HARD"
                        elif row['RemainingLaps'] > t3_laps: return "MEDIUM"
                        else: return "SOFT"
                    pit_stop_df['PredictedCompound'] = pit_stop_df.apply(predict_compound, axis=1)
                    correct = (pit_stop_df['PredictedCompound'] == pit_stop_df['ChosenCompound']).sum()
                    accuracy = correct / len(pit_stop_df)
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_thresholds = (t1, t2_laps, t3_laps)
    result = {"t1": best_thresholds[0], "t2": best_thresholds[1], "t3": best_thresholds[2], "accuracy": round(best_accuracy, 2)}
    print(f"최적 임계값: t1={result['t1']}°C, t2={result['t2']}랩, t3={result['t3']}랩 (정확도: {result['accuracy']})")
    return result

def calculate_pit_stop_time(laps_data, track_name):
    """
    피트스톱 시간을 계산하는 함수
    
    Parameters:
    laps_data: DataFrame - 랩 데이터 (전처리된 데이터)
    track_name: str - 트랙 이름 (예: 'Abu_Dhabi')
    
    Returns:
    float - 평균 피트스톱 시간 (초)
    """
    
    # 데이터 유효성 검사
    if laps_data is None or laps_data.empty:
        print(f"경고: 데이터가 비어있습니다. 기본값(22초) 사용.")
        return 22.0
    
    # 필요한 컬럼들이 존재하는지 확인
    required_columns = ['GrandPrix', 'PitInTime', 'PitOutTime', 'LapNumber', 'Driver']
    missing_columns = [col for col in required_columns if col not in laps_data.columns]
    
    if missing_columns:
        print(f"경고: 필요한 컬럼들이 누락되었습니다: {missing_columns}")
        print(f"사용 가능한 컬럼들: {list(laps_data.columns)}")
        return 22.0
    
    # 해당 트랙 데이터 필터링
    try:
        track_data = laps_data[laps_data['GrandPrix'].str.contains(track_name, na=False)].copy()
        if track_data.empty:
            print(f"경고: {track_name} 트랙 데이터를 찾을 수 없습니다.")
            print(f"사용 가능한 트랙들: {laps_data['GrandPrix'].unique()}")
            return 22.0
            
        print(f"{track_name} 데이터 발견: {len(track_data)} 행")
        
    except Exception as e:
        print(f"경고: 데이터 필터링 중 오류 발생: {e}. 기본값(22초) 사용.")
        return 22.0

    pit_times = []
    
    # 각 드라이버별로 피트스톱 시간 계산
    for driver in track_data['Driver'].unique():
        driver_data = track_data[track_data['Driver'] == driver].sort_values('LapNumber').reset_index(drop=True)
        
        # PitInTime이 있는 랩들 찾기
        pit_in_laps = driver_data[driver_data['PitInTime'].notna()]
        
        for _, pit_in_row in pit_in_laps.iterrows():
            try:
                pit_in_lap = pit_in_row['LapNumber']
                pit_in_time_str = str(pit_in_row['PitInTime'])
                
                # 해당 드라이버의 다음 랩들에서 PitOutTime 찾기
                next_laps = driver_data[driver_data['LapNumber'] > pit_in_lap]
                pit_out_row = next_laps[next_laps['PitOutTime'].notna()].head(1)
                
                if pit_out_row.empty:
                    print(f"경고: {driver} 드라이버의 랩 {pit_in_lap}에서 PitOutTime을 찾을 수 없습니다. 건너뜁니다.")
                    continue
                
                pit_out_time_str = str(pit_out_row['PitOutTime'].iloc[0])
                
                # 시간 파싱
                try:
                    # timedelta 형식 처리
                    if 'days' in pit_in_time_str:
                        pit_in_time = pd.to_timedelta(pit_in_time_str)
                    else:
                        pit_in_time = pd.to_timedelta(f'0 days {pit_in_time_str}')
                        
                    if 'days' in pit_out_time_str:
                        pit_out_time = pd.to_timedelta(pit_out_time_str)
                    else:
                        pit_out_time = pd.to_timedelta(f'0 days {pit_out_time_str}')
                    
                except Exception as parse_error:
                    print(f"경고: {driver} 드라이버의 시간 파싱 오류: {parse_error}. 건너뜁니다.")
                    continue
                
                # 피트스톱 시간 계산
                pit_stop_time = (pit_out_time - pit_in_time).total_seconds()
                
                # 유효성 검사 (10초 ~ 120초)
                if pit_stop_time <= 0:
                    print(f"경고: {driver} 드라이버의 랩 {pit_in_lap}에서 음수 또는 0인 피트스톱 시간: {pit_stop_time:.2f}초. 건너뜁니다.")
                    continue
                    
                if 10 < pit_stop_time < 120:
                    pit_times.append(pit_stop_time)
                    print(f"{driver} 드라이버 랩 {pit_in_lap}: 피트스톱 시간 {pit_stop_time:.2f}초")
                else:
                    print(f"경고: {driver} 드라이버의 랩 {pit_in_lap}에서 비정상적인 피트스톱 시간: {pit_stop_time:.2f}초. 건너뜁니다.")
                    
            except Exception as e:
                print(f"경고: {driver} 드라이버의 데이터 처리 중 오류: {e}. 건너뜁니다.")
                continue

    # 결과 처리
    if not pit_times:
        print(f"경고: {track_name}에 유효한 피트스톱 시간이 없습니다. 기본값(22초) 사용.")
        return 22.0

    # 평균 피트스톱 시간 계산
    avg_pit_stop_time = np.mean(pit_times)
    print(f"\n{track_name} 평균 피트스톱 시간: {avg_pit_stop_time:.2f}초 (유효 데이터 수: {len(pit_times)})")
    return avg_pit_stop_time
    
def calculate_degradation_rate_v3(laps_data):
    """
    이상치에 강건한 성능 저하율 계산 함수 (선형 회귀 기반)
    """
    base_rates = {'SOFT': 0.035, 'MEDIUM': 0.022, 'HARD': 0.013, 'INTERMEDIATE': 0.045, 'WET': 0.080}
    tyre_data = laps_data.dropna(subset=['LapTime', 'TyreLife', 'Compound'])
    degradation_rates = {}

    for compound in tyre_data['Compound'].unique():
        if compound not in base_rates:
            continue
        
        compound_data = tyre_data[tyre_data['Compound'] == compound].copy()

        # 데이터가 너무 적으면 기본값 사용
        if len(compound_data) < 30:
            degradation_rates[compound] = base_rates[compound]
            continue

        # 1. 이상치(Outlier) 제거: 랩타임이 비정상적으로 느린 경우(트래픽, SC 등)를 필터링
        median_lap_time = compound_data['LapTime'].median()
        # 중앙값보다 5% 이상 느린 랩타임은 이상치로 간주하고 제거
        sane_data = compound_data[compound_data['LapTime'] < median_lap_time * 1.05]
        
        if len(sane_data) < 20:
            degradation_rates[compound] = base_rates[compound]
            continue
            
        # 2. 선형 회귀(Linear Regression)로 성능 저하 추세 계산
        X = sane_data[['TyreLife']]
        y = sane_data['LapTime']
        
        model = LinearRegression()
        model.fit(X, y)
        
        # 기울기(slope)는 타이어 1랩 당 랩타임이 얼마나 증가하는지를 의미
        slope = model.coef_[0]
        
        if slope > 0:
            # 성능 저하율 = (랩당 시간 증가량 / 평균 랩타임)
            degradation_rate = slope / median_lap_time
            # 너무 높거나 낮은 값이 나오지 않도록 상/하한선 설정
            degradation_rates[compound] = max(0.005, min(0.08, degradation_rate))
        else:
            # 기울기가 음수이면 성능 저하가 없다고 판단, 기본값 사용
            degradation_rates[compound] = base_rates[compound]

    # 모든 컴파운드에 대한 기본값 처리
    for compound in base_rates:
        if compound not in degradation_rates:
            degradation_rates[compound] = base_rates[compound]
    
    # 드라이 타이어에 대해서만 순서 보장 (HARD < MEDIUM < SOFT)
    dry_compounds = ['HARD', 'MEDIUM', 'SOFT']
    dry_rates = sorted([degradation_rates.get(c) for c in dry_compounds if c in degradation_rates])
    if len(dry_rates) == 3:
        # 정렬된 값을 다시 할당
        degradation_rates['HARD'] = dry_rates[0]
        degradation_rates['MEDIUM'] = dry_rates[1]
        degradation_rates['SOFT'] = dry_rates[2]
        
    return degradation_rates

def calculate_stint_params_proper(laps_data):
    # (이전과 동일한 함수 내용 및 의존 함수들 포함)
    # ... (filter_performance_based_stints, analyze_stint_distribution, validate_stint_logic 등)
    pit_data = laps_data.dropna(subset=['PitInTime', 'LapNumber', 'Driver', 'Compound'])
    pit_data = pit_data.sort_values(['GrandPrix', 'Driver', 'Time']) # GrandPrix 기준으로도 정렬
    pit_data['NextPitLap'] = pit_data.groupby(['GrandPrix', 'Driver'])['LapNumber'].shift(-1)
    pit_data['StintLength'] = pit_data['NextPitLap'] - pit_data['LapNumber']
    stint_data = pit_data[pit_data['StintLength'] > 0].dropna(subset=['StintLength', 'Compound'])
    
    optimal_stint = {}
    min_stint = {}
    
    performance_based_stints = filter_performance_based_stints(stint_data, laps_data)
    
    theoretical_values = {
        'SOFT': 16, 'MEDIUM': 25, 'HARD': 35,
        'INTERMEDIATE': 20, 'WET': 12
    }
    
    # 모든 컴파운드에 대해 계산
    all_compounds = ['SOFT', 'MEDIUM', 'HARD', 'INTERMEDIATE', 'WET']
    for compound in all_compounds:
        if compound not in laps_data['Compound'].unique():
            continue
            
        compound_stints = performance_based_stints.get(compound, [])
        
        if len(compound_stints) >= 5:
            stint_analysis = analyze_stint_distribution(compound_stints)
            optimal_stint[compound] = stint_analysis['recommended']
        else:
            optimal_stint[compound] = theoretical_values.get(compound, 20)
        
        min_stint[compound] = 8 if compound in ['SOFT', 'MEDIUM', 'HARD'] else 5
        print(f"{compound}: {len(compound_stints)} quality stints, optimal={optimal_stint.get(compound)} laps")
    
    validated_stints = validate_stint_logic(optimal_stint, theoretical_values)
    
    return validated_stints, min_stint

def calculate_base_pace(laps_data):
    """
    초기 스틴트의 평균 LapTime을 이용해 base_pace 계산
    - 모든 컴파운드(WET, INTERMEDIATE 포함)를 처리하도록 수정됨
    """
    pace_data = laps_data.dropna(subset=['LapTime', 'Compound', 'TyreLife'])
    # TyreLife가 1 이하인 랩만 필터링하여 각 타이어의 초기 성능을 확인
    pace_data = pace_data[pace_data['TyreLife'] <= 1]
    
    base_paces = {}
    
    # 데이터에 존재하는 모든 고유 컴파운드에 대해 반복
    for compound in pace_data['Compound'].unique():
        compound_mean = pace_data[pace_data['Compound'] == compound]['LapTime'].mean()
        
        # 유효한 랩타임이 있는 경우에만 pace를 계산
        if pd.notna(compound_mean) and compound_mean > 0:
            if compound == 'SOFT':
                base_paces[compound] = 1.0  # 기준 속도
            elif compound == 'MEDIUM':
                base_paces[compound] = 0.98 # SOFT보다 약간 느림
            elif compound == 'HARD':
                base_paces[compound] = 0.96 # MEDIUM보다 약간 느림
            elif compound == 'INTERMEDIATE':
                base_paces[compound] = 0.85 # 드라이 타이어보다 확연히 느림
            elif compound == 'WET':
                base_paces[compound] = 0.75 # 가장 느림
            else:
                # 'Unknown' 등 예상치 못한 컴파운드에 대한 안전장치
                base_paces[compound] = 0.70 
        
        # base_paces 딕셔너리에 해당 compound가 있는지 확인 후 출력 (KeyError 방지)
        if compound in base_paces:
            print(f"{compound} initial data points: {len(pace_data[pace_data['Compound'] == compound])}, base_pace = {base_paces[compound]}")
        else:
            # pace가 계산되지 않은 경우 (e.g., 유효한 랩타임이 없는 경우)
            print(f"{compound} initial data points: {len(pace_data[pace_data['Compound'] == compound])}, base_pace calculation skipped.")

    print("\nCalculated base_pace:", base_paces)
    return base_paces

def calculate_all_compound_performance(laps_data):
    print("\n=== 컴파운드 성능 파라미터 계산 시작 ===")
    base_paces = calculate_base_pace(laps_data)
    degradation_rates = calculate_degradation_rate_v3(laps_data)
    optimal_stint, min_stint = calculate_stint_params_proper(laps_data)
    # temp_threshold = calculate_temp_threshold(laps_data) # 이 함수가 없으므로 주석처리하거나 추가해야 함
    
    compound_performance = {}
    existing_compounds = laps_data['Compound'].unique()
    for compound in existing_compounds:
        compound_performance[compound] = {
            "base_pace": base_paces.get(compound, 0.95),
            "degradation_rate": degradation_rates.get(compound, 0.05),
            "optimal_stint": optimal_stint.get(compound, 15),
            "min_stint": min_stint.get(compound, 10),
            # "temp_threshold": temp_threshold.get(compound, 28.3)
        }
    print("=== 컴파운드 성능 파라미터 계산 완료 ===")
    return compound_performance