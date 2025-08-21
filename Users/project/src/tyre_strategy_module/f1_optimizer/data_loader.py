# f1_optimizer/data_loader.py
import pandas as pd
import json

# AzureStorageAccess 클래스를 임포트
from Users.project.src.data_container.data_container import AzureStorageAccess
from .config import COMPOUND_MAP, CONFIG_FILE_PATH

def load_and_merge_data(track_name):
    """
    AzureStorageAccess 클래스를 사용하여 Azure Blob에서 데이터를 로드하고
    초기 병합 및 시간 변환을 수행합니다.
    """
    with open(CONFIG_FILE_PATH, 'r', encoding='utf-8') as f:
        tracks_config = json.load(f)
    if track_name not in tracks_config:
        raise ValueError(f"'{track_name}' 설정이 '{CONFIG_FILE_PATH}'에 없습니다.")

    track_info = tracks_config[track_name]
    race_laps = track_info['total_laps']
    target_group = track_info.get('group')

    grand_prix_list = []
    if target_group:
        print(f"'{track_name}' 트랙은 '{target_group}' 그룹에 속합니다. 그룹 전체 데이터를 로드합니다.")
        for t_name, t_info in tracks_config.items():
            if t_info.get('group') == target_group:
                grand_prix_list.extend(t_info['data_files'])
    else:
        print(f"'{track_name}' 트랙에 그룹이 지정되지 않았습니다. 단일 트랙 데이터만 로드합니다.")
        grand_prix_list = track_info['data_files']

    print(f"데이터 로딩 대상: {grand_prix_list}")

    # AzureStorageAccess 인스턴스 생성
    azure_access = AzureStorageAccess()

    all_laps_data = []
    all_weather_data = []

    for gp in grand_prix_list:
        session_info = azure_access.read_csv_by_data_frame(f'{gp}/session_info.csv')
        base_time = None
        if session_info is not None and not session_info.empty:
            race_session = session_info[session_info['Type'] == 'Race']
            if not race_session.empty and 'StartDate' in race_session.columns:
                base_time = pd.to_datetime(race_session['StartDate'].iloc[0])
        
        if base_time is None:
            year = gp.split('/')[0]
            base_time = pd.to_datetime(f'{year}-01-01 14:00:00')
            print(f"Warning: Failed to load valid StartDate for {gp}, using default base_time: {base_time}")

        laps_data = azure_access.read_csv_by_data_frame(f'{gp}/laps.csv')
        if laps_data is not None:
            laps_data['Time'] = pd.to_timedelta(laps_data['Time'].astype(str).str.replace('0 days', '').str.strip())
            laps_data['Time'] = base_time + laps_data['Time']
            laps_data['GrandPrix'] = gp
            if 'Compound' in laps_data.columns:
                laps_data['Compound'] = laps_data['Compound'].replace(COMPOUND_MAP)
                drop_compounds = ['TEST', 'TEST_UNKNOWN', 'UNKNOWN']
                laps_data = laps_data[~laps_data['Compound'].isin(drop_compounds)].reset_index(drop=True)
            all_laps_data.append(laps_data)
        
        weather_data = azure_access.read_csv_by_data_frame(f'{gp}/weather_data.csv')
        if weather_data is not None:
            weather_data['Time'] = pd.to_timedelta(weather_data['Time'].astype(str).str.replace('0 days', '').str.strip())
            weather_data['Time'] = base_time + weather_data['Time']
            weather_data['GrandPrix'] = gp
            weather_data = weather_data.sort_values('Time')
            all_weather_data.append(weather_data)

    if not all_laps_data:
        raise ValueError("No valid lap data could be loaded. Aborting.")
        
    laps_df = pd.concat(all_laps_data, ignore_index=True)
    
    weather_df = pd.DataFrame()
    if all_weather_data:
        weather_df = pd.concat(all_weather_data, ignore_index=True)

    # Unknown 컴파운드 제거
    if 'Compound' in laps_df.columns:
        laps_df = laps_df[laps_df['Compound'] != 'Unknown'].reset_index(drop=True)

    return laps_df, weather_df, race_laps