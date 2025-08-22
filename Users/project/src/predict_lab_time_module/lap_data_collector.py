import pandas as pd
import numpy as np

def get_weather_data(current_time, data_frame: pd.DataFrame):
    last_data = data_frame[data_frame["Time"] <= current_time].tail(1)
    return last_data

def get_car_data_feature(data_frame: pd.DataFrame) -> list[float]:
    # 데이터 추출
    rpm_datas = data_frame["RPM"]
    speed_datas = data_frame["Speed"]
    gear_datas = data_frame["nGear"]
    throttle_datas = data_frame["Throttle"]
    brake_datas = data_frame["Brake"]
    drs_datas = data_frame["DRS"]
    # 피처 리스트 초기화
    features = []

    # 1. RPM 피처
    features.append(rpm_datas.mean())  # 평균 RPM
    features.append(rpm_datas.max())   # 최대 RPM
    features.append(rpm_datas.diff().abs().mean())  
    features.append(rpm_datas.std())   # RPM 표준편차 (변동성)
    # 2. Speed 피처
    features.append(speed_datas.mean())  # 평균 속도
    features.append(speed_datas.max())   # 최대 속도
    features.append(speed_datas.diff().abs().mean())  # 평균 속도 변화율 (가속도)
    features.append(speed_datas.std())   # 속도 표준편차
    # 3. nGear 피처
    features.append(gear_datas.mean()) 
    features.append(gear_datas.max())  
    features.append(gear_datas.diff().abs().mean())  # 평균 기어 변화율
    features.append(gear_datas.std()) 
    features.append(gear_datas.value_counts().max())  # 가장 많이 사용된 기어의 빈도
    features.append((gear_datas.shift() != gear_datas)[1:].sum())  # 기어 전환 횟수
    # 4. Throttle 피처
    features.append(throttle_datas.mean())  # 평균 스로틀 사용
    features.append(throttle_datas.max())   
    features.append(throttle_datas.diff().abs().mean()) 
    features.append(throttle_datas.std()) 
    features.append((throttle_datas > 0.95).sum() / len(throttle_datas)) # 거의 풀 스로틀의 비율
    # 5. Brake 피처
    features.append(brake_datas.mean())  # 브레이크 사용 비율
    # 6. DRS 피처
    features.append(drs_datas.mean())  # DRS 평균
    features.append(drs_datas.max())  # DRS 최대
    features.append(drs_datas.diff().abs().mean())  # 평균 drs 변화율
    features.append(drs_datas.std())  
    features.append(drs_datas.value_counts().max())  # 가장 많이 사용된 drs 빈도
    features.append((drs_datas.shift() != drs_datas)[1:].sum())  # drs 전환 횟수
#       코스팅(Coasting) 시간 비율: 스로틀과 브레이크를 모두 사용하지 않는 타력 주행 구간의 비율입니다. 드라이버의 효율성을 나타내는 지표가 될 수 있습니다.
#       # Throttle과 Brake가 모두 5% 미만인 시간의 비율
#       ((throttle_datas < 0.05) & (brake_datas < 0.05)).sum() / len(data_frame)
    return features

def get_laps_data_feature(row: pd.DataFrame) -> list[float]:
    stint = row["Stint"]
    tyre_life = row.TyreLife
    lap_number = row.LapNumber
    # sector1_time = pd.to_timedelta(row["Sector1Time"]).iloc[0].total_seconds()
    # sector2_time = pd.to_timedelta(row["Sector2Time"]).iloc[0].total_seconds()
    # sector3_time = pd.to_timedelta(row["Sector3Time"]).iloc[0].total_seconds()
    sector1_time = pd.to_timedelta(row.Sector1Time).total_seconds()
    sector2_time = pd.to_timedelta(row.Sector2Time).total_seconds()
    sector3_time = pd.to_timedelta(row.Sector3Time).total_seconds()
    fresh_tyre = row.FreshTyre
    track_status =  row["TrackStatus"]
    features = []
    # 타이어 상태 원핫인코딩
    compound = row["Compound"]
    tyre_mapping = {"SOFT": [1, 0, 0], "MEDIUM": [0, 1, 0], "HARD": [0, 0, 1]}
    compound_encoded = tyre_mapping.get(compound, [1, 0, 0])  # 알 수 없는 타이어는 [0,0,0]
    track_flags = []
    flag = 16  # 2^4부터 시작 (5비트)
    while flag != 0:
        track_flags.append(1.0 if track_status & flag else 0.0)
        flag //= 2
    features.append(fresh_tyre)
    features.append(stint)
    features.append(tyre_life)
    features.append(lap_number)
    features.append(sector1_time)
    features.append(sector2_time)
    features.append(sector3_time)
    
    features += compound_encoded
    #features += track_flags
    return features

def get_weather_data_feature(data_frame: pd.DataFrame) -> list[float]:
    air_temp = data_frame["AirTemp"].item()
    humidity = data_frame["Humidity"].item()
    pressure = data_frame["Pressure"].item()
    rainfall = data_frame["Rainfall"].item()
    track_temp = data_frame["TrackTemp"].item()
    wind_direction = data_frame["WindDirection"].item()
    wind_speed = data_frame["WindSpeed"].item()
    features = [air_temp, humidity, pressure, rainfall, track_temp, wind_direction, wind_speed]
    return features

class LapDataCollector:
    def __init__(self) -> None:
        self.car_data_frame = pd.DataFrame()
        
    def add_car_data(self, car_data_frame: pd.DataFrame):
        if self.car_data_frame.empty:
            self.car_data_frame = car_data_frame
        else:
            self.car_data_frame = pd.concat([self.car_data_frame, car_data_frame], ignore_index=True)
        
    def get_feature_by_data_frame(self, laps_data_frame: pd.DataFrame, weather_data_frame: pd.DataFrame) -> pd.DataFrame:
        car_data_feature = get_car_data_feature(self.car_data_frame)
        laps_data_feature = get_laps_data_feature(laps_data_frame)
        weather_data_feature = get_weather_data_feature(weather_data_frame)
        one_lap_features = []
        one_lap_features += car_data_feature
        one_lap_features += laps_data_feature
        one_lap_features += weather_data_feature
        one_lap_features = [float(x) if isinstance(x, (np.float32, np.float64)) else x for x in one_lap_features]
        one_lap_features = [int(x) if isinstance(x, (np.int32, np.int64)) else x for x in one_lap_features]
        
        return pd.DataFrame([one_lap_features])

    def clear_car_data(self):
        self.car_data_frame = pd.DataFrame()
    
    
def arrange_feature_and_label_has_nan(car_data: pd.DataFrame, laps_data: pd.DataFrame, weather_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    car_data["Time"] = pd.to_timedelta(car_data["Time"])
    laps_data["LapTime"] = pd.to_timedelta(laps_data["LapTime"])
    weather_data["Time"] = pd.to_timedelta(weather_data["Time"])
    progress_time = pd.Timedelta(0)
    one_team_features = []
    one_team_label = []
    for row in laps_data.itertuples():
        lap_number = row.LapNumber
        is_accurate = row.IsAccurate
        if is_accurate == False:
            continue
        track_status = row.TrackStatus
        if track_status != 1:
            continue
        lap_time = row.LapTime
        progress_time += lap_time
        same_lap_number_data = car_data[car_data["LapNumber"] == lap_number]
        if same_lap_number_data.empty or len(same_lap_number_data) < 2:
            continue
        weather_frame = get_weather_data(progress_time, weather_data)
        if weather_frame.empty:
            continue
        one_lap_features = []
        car_data_feature = get_car_data_feature(same_lap_number_data)
        laps_data_feature = get_laps_data_feature(row)
        weather_data_feature = get_weather_data_feature(weather_frame)
        one_lap_features += car_data_feature
        one_lap_features += laps_data_feature
        one_lap_features += weather_data_feature
        one_lap_features = [float(x) if isinstance(x, (np.float32, np.float64)) else x for x in one_lap_features]
        one_lap_features = [int(x) if isinstance(x, (np.int32, np.int64)) else x for x in one_lap_features]
        one_team_features.append(one_lap_features)
        one_team_label.append([lap_time.total_seconds()])  
    
    feature_headers = [
        # 1. RPM
        "rpm_mean", "rpm_max", "rpm_diff_mean", "rpm_std",
        # 2. Speed
        "speed_mean", "speed_max", "speed_diff_mean", "speed_std",
        # 3. nGear
        "gear_mean", "gear_max", "gear_diff_mean", "gear_std",
        "gear_most_freq", "gear_change_count",
        # 4. Throttle
        "throttle_mean", "throttle_max", "throttle_diff_mean", "throttle_std",
        "throttle_full_ratio",
        # 5. Brake
        "brake_mean",
        # 6. DRS
        "drs_mean", "drs_max", "drs_diff_mean", "drs_std",
        "drs_most_freq", "drs_change_count",
        # Laps Feature
        "fresh_tyre",
        "stint",
        "tyre_life",
        "lap_number",
        "sector1_time", "sector2_time", "sector3_time",
        "tyre_SOFT", "tyre_MEDIUM", "tyre_HARD",
        #"flag_16", "flag_8", "flag_4", "flag_2", "flag_1",  # 5비트
        # Weather Feature
        "air_temp", "humidity", "pressure", "rainfall", "track_temp",
        "wind_direction", "wind_speed"
    ]
    one_team_features = pd.DataFrame(one_team_features, columns=feature_headers)
    # 결측값 채우기 얘네는 채워도됌
    one_team_features["stint"] = one_team_features["stint"].ffill()
    one_team_features["tyre_life"] = one_team_features["tyre_life"].ffill()
    one_team_label = pd.DataFrame(one_team_label)
    return one_team_features, one_team_label