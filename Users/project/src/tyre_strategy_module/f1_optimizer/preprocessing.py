# f1_optimizer/preprocessing.py
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def fix_time_sorting(df, time_col='Time', group_col='GrandPrix'):
    """
    다양한 시간 형식을 처리하고 정렬 문제를 해결하는 안정적인 함수.
    """
    df = df.copy() # 원본 수정을 방지하기 위해 복사본으로 시작

    # [핵심 수정] to_datetime이 실패하더라도 NaT로 만들고, dropna는 맨 나중에 수행
    # errors='coerce'는 변환에 실패한 값을 NaT(Not a Time)로 만듭니다.
    df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
    
    # NaT가 아닌 유효한 시간 데이터만 남김
    df.dropna(subset=[time_col], inplace=True)
    if df.empty:
        return pd.DataFrame() # 유효한 시간 데이터가 없으면 빈 프레임 반환

    fixed_groups = []
    for group_name in df[group_col].unique():
        group_data = df[df[group_col] == group_name].copy().sort_values(time_col).reset_index(drop=True)
        
        # is_monotonic_increasing 대신 diff()를 사용하여 시간 역전 현상을 더 안정적으로 감지
        time_diffs = group_data[time_col].diff().dt.total_seconds()
        if (time_diffs < 0).any():
            print(f"Warning: {group_name} has non-monotonic time values, fixing...")
            # 중복 또는 역전된 시간에만 마이크로초를 더해 순서를 보장
            group_data = group_data.sort_values(time_col, kind='mergesort').reset_index(drop=True)
        
        fixed_groups.append(group_data)
        
    return pd.concat(fixed_groups, ignore_index=True)

def merge_weather_data(laps_data, weather_data):
    if not weather_data.empty:
        merged_groups = []
        for gp in laps_data['GrandPrix'].unique():
            laps_gp = laps_data[laps_data['GrandPrix'] == gp].copy()
            weather_gp = weather_data[weather_data['GrandPrix'] == gp].copy()
            if weather_gp.empty:
                laps_gp['TrackTemp'] = weather_data['TrackTemp'].mean()
                laps_gp['AirTemp'] = weather_data.get('AirTemp', pd.Series([25.0])).mean()
                laps_gp['Rainfall'] = 0
                merged_groups.append(laps_gp)
                continue
            laps_gp = laps_gp.sort_values('Time').reset_index(drop=True)
            weather_gp = weather_gp.sort_values('Time').reset_index(drop=True)
            merged_gp = pd.merge_asof(
                laps_gp, weather_gp.drop(columns=['GrandPrix'], errors='ignore'),
                on='Time', direction='nearest', tolerance=pd.Timedelta(minutes=20)
            )
            merged_groups.append(merged_gp)
        if merged_groups:
            laps_data = pd.concat(merged_groups, ignore_index=True)
    else:
        laps_data['TrackTemp'] = 31.2
        laps_data['AirTemp'] = 25.0
        laps_data['Rainfall'] = 0
    return laps_data

def calculate_stints(df):
    df = df.copy()
    df = df.sort_values(['GrandPrix', 'Driver', 'Time'])
    df['IsPitStop'] = df['PitInTime'].notna()
    df['Stint'] = df.groupby(['GrandPrix', 'Driver'])['IsPitStop'].cumsum() + 1
    
    pit_laps = df.loc[df['IsPitStop'], ['GrandPrix', 'Driver', 'LapNumber']].copy()
    pit_laps['NextPitLap'] = pit_laps.groupby(['GrandPrix', 'Driver'])['LapNumber'].shift(-1)
    
    df = pd.merge(df, pit_laps[['GrandPrix', 'Driver', 'LapNumber', 'NextPitLap']], 
                  on=['GrandPrix', 'Driver', 'LapNumber'], how='left')
    
    df['NextPitLap'] = df.groupby(['GrandPrix', 'Driver', 'Stint'])['NextPitLap'].bfill()
    
    race_laps = df.groupby(['GrandPrix'])['LapNumber'].max()
    df['RaceMaxLap'] = df['GrandPrix'].map(race_laps)
    # [FutureWarning 해결] .fillna()의 inplace=True 대신, 값을 직접 할당
    df['NextPitLap'] = df['NextPitLap'].fillna(df['RaceMaxLap'])
    
    df['StintStartLap'] = df.groupby(['GrandPrix', 'Driver', 'Stint'])['LapNumber'].transform('first')
    df['StintLength'] = df['NextPitLap'] - df['StintStartLap']
    
    df.drop(columns=['RaceMaxLap', 'StintStartLap', 'NextPitLap'], inplace=True)
    return df

def feature_engineer(laps_data):
    laps_data = laps_data.copy()

    if 'IsPitStop' not in laps_data.columns:
        laps_data['IsPitStop'] = laps_data['PitInTime'].notna().astype('Int64')

    boolean_cols = [col for col in ['IsPersonalBest', 'FreshTyre', 'Deleted', 'FastF1Generated', 'IsAccurate', 'Rainfall'] if col in laps_data.columns]
    for col in boolean_cols:
        laps_data[col] = pd.to_numeric(laps_data[col], errors='coerce').fillna(0).astype('Int64')

    time_cols = [col for col in ['LapTime', 'Sector1Time', 'Sector2Time', 'Sector3Time'] if col in laps_data.columns]
    for col in time_cols:
        if laps_data[col].dtype != 'float64':
            laps_data[col] = pd.to_timedelta(laps_data[col], errors='coerce').dt.total_seconds()
    
    laps_data.dropna(subset=['LapTime', 'Compound', 'TyreLife'], inplace=True)

    speed_cols = [col for col in ['SpeedI1', 'SpeedFL', 'SpeedI2', 'SpeedST'] if col in laps_data.columns]
    for col in speed_cols:
        laps_data[col] = laps_data[col].interpolate(method='linear').fillna(laps_data[col].median())

    if all(col in laps_data.columns for col in ['GrandPrix', 'Driver', 'Stint', 'TyreLife']):
        laps_data['TyreLife'] = laps_data.groupby(['GrandPrix', 'Driver', 'Stint'])['TyreLife'].transform(lambda x: x.fillna(x.median()))
        laps_data['TyreWear'] = laps_data.groupby(['GrandPrix', 'Driver', 'Stint'])['TyreLife'].transform('max') - laps_data['TyreLife']
    
    if all(col in laps_data.columns for col in ['GrandPrix', 'Driver', 'Stint', 'LapTime']):
        laps_data['LapTime_Delta'] = laps_data.groupby(['GrandPrix', 'Driver', 'Stint'])['LapTime'].diff().fillna(0)
    
    if all(col in laps_data.columns for col in ['GrandPrix', 'Driver', 'Stint', 'SpeedI1']):
        laps_data['SpeedI1_Delta'] = laps_data.groupby(['GrandPrix', 'Driver', 'Stint'])['SpeedI1'].diff().fillna(0)
    
    if all(col in laps_data.columns for col in ['GrandPrix', 'Driver', 'Stint', 'TyreWear', 'LapNumber']) and not laps_data.empty:
        laps_data['TyreWear_Rate'] = (laps_data.groupby(['GrandPrix', 'Driver', 'Stint'])['TyreWear'].diff().fillna(0) / laps_data['LapNumber'].replace(0, 1)).replace([np.inf, -np.inf], 0)

    return laps_data

def get_preprocessor(laps_data):
    numerical_cols_candidate = [
        'LapNumber', 'Stint', 'SpeedI1', 'SpeedI2', 'SpeedFL', 'SpeedST', 'TyreLife', 
        'Position', 'TrackTemp', 'AirTemp', 'Humidity', 'Pressure', 'WindDirection', 
        'WindSpeed', 'LapTime_Delta', 'SpeedI1_Delta', 'TyreWear_Rate', 'TyreWear', 'Rainfall'
    ]
    categorical_cols_candidate = ['Compound', 'TrackStatus', 'Driver', 'Team', 'DriverNumber']

    numerical_cols = [col for col in numerical_cols_candidate if col in laps_data.columns and pd.api.types.is_numeric_dtype(laps_data[col])]
    categorical_cols = [col for col in categorical_cols_candidate if col in laps_data.columns]
    
    print(f"Numerical columns for preprocessing: {numerical_cols}")
    print(f"Categorical columns for preprocessing: {categorical_cols}")

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numerical_cols),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ]), categorical_cols)
        ],
        remainder='drop'
    )
    return preprocessor, numerical_cols, categorical_cols