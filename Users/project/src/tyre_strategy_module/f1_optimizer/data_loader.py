
track = [
    "Eifel",
    "Emilia_Romagna",
    "French",
    "German",
    "Hungarian",
    "Italian",
    "Japanese",
    "Las_Vegas",
    "Mexico_City",
    "Miami",
    "Monaco",
    "Portuguese",
    "Qatar",
    "Russian",
    "Sakhir",
    "Saudi_Arabian",
    "Singapore",
    "Spanish",
    "Styrian",
    "Sao_Paulo",
    "Turkish",
    "Tuscan",
    "United_States"
]
compound = ["HARD", "MEDIUM", "SOFT"]

# track_name = 'Australian' # <--- 분석하고 싶은 트랙 이름만 여기서 변경하세요.
# initial_compound = "D" # 시뮬레이션 시작 타이어 (필요시 변경)

for track_name in track :


    print(f">>> STEP 3: '{track_name}' 데이터 및 파라미터 로딩 시작...")

    # 트랙 설정 파일 로드
    with open('/home/azureuser/cloudfiles/code/Users/data/tracks_config.json', 'r', encoding='utf-8') as f:
        tracks_config = json.load(f)
    if track_name not in tracks_config:
        raise ValueError(f"'{track_name}' 설정이 'tracks_config.json'에 없습니다.")

    # 타겟 트랙의 정보 추출
    track_info = tracks_config[track_name]
    race_laps = track_info['total_laps']
    target_group = track_info.get('group') # 타겟 트랙의 그룹 이름 가져오기

    # grand_prix_list 생성
    grand_prix_list = []
    if target_group:
        print(f"'{track_name}' 트랙은 '{target_group}' 그룹에 속합니다. 그룹 전체 데이터를 로드합니다.")
        # 전체 설정 파일에서 같은 그룹에 속한 모든 트랙을 찾음
        for t_name, t_info in tracks_config.items():
            if t_info.get('group') == target_group:
                grand_prix_list.extend(t_info['data_files']) # 해당 트랙의 데이터 파일 목록을 추가
    else:
        # 그룹이 지정되지 않은 경우, 해당 트랙의 데이터만 사용
        print(f"'{track_name}' 트랙에 그룹이 지정되지 않았습니다. 단일 트랙 데이터만 로드합니다.")
        grand_prix_list = track_info['data_files']

    print(f"데이터 로딩 대상: {grand_prix_list}")


    # Azure Blob Storage 설정
    azure_access = AzureStorageAccess()
    # 모든 파일 순회하고 싶다.
    # for file in azure_access.get_all_file():
    #     file_name = file.name
    #     data_frame = azure_access.get_file_by_data_frame(file_name)   <- 지금 돌고있는 파일블롭을 바로 데이터로 변환

    #       data_frame = azure_access.read_csv_by_data_frame(file_name) <- 지금 돌고있는 파일블롭의 이름을 뽑아서 다시 파일 전부 돌아가며 이름과 똑같은 파일찾아서 반환


    all_laps_data = []
    all_weather_data = []

    compound_map = {
        'SOFT': 'SOFT',
        'SUPERSOFT': 'SOFT',
        'ULTRASOFT': 'SOFT',
        'HYPERSOFT': 'SOFT',
        'MEDIUM': 'MEDIUM',
        'HARD': 'HARD',
        'INTERMEDIATE': 'INTERMEDIATE',   # 비건조
        'WET': 'WET',                     # 비건조
        'TEST': 'TEST',
        'TEST_UNKNOWN': 'Unknown',
        'UNKNOWN': 'Unknown'
    }

    for gp in grand_prix_list:
        session_info = azure_access.read_csv_by_data_frame(f'{gp}/session_info.csv')
        if session_info is not None:
            print(f"Loaded {gp}/session_info.csv with columns: {session_info.columns.tolist()}")
            base_time = pd.to_datetime(session_info[session_info['Type'] == 'Race']['StartDate'].iloc[0])
        else:
            year = gp.split('/')[0]
            base_time = pd.to_datetime(f'{year}-12-08 14:00:00')
            print(f"Warning: Failed to load {gp}/session_info.csv, using default base_time: {base_time}")
        
        laps_data = azure_access.read_csv_by_data_frame(f'{gp}/laps.csv')
        if laps_data is not None:
            print(f"Loaded {gp}/laps.csv with columns: {laps_data.columns.tolist()}")
            laps_data['Time'] = pd.to_timedelta(laps_data['Time'].astype(str).str.replace('0 days', '').str.strip())
            laps_data['Time'] = base_time + laps_data['Time']
            laps_data['GrandPrix'] = gp
            if 'Compound' in laps_data.columns:
                laps_data['Compound'] = laps_data['Compound'].replace(compound_map)
                # 제거할 Compound 목록
                drop_compounds = ['TEST', 'TEST_UNKNOWN', 'UNKNOWN']
                # 해당 Compound인 행 제거
                laps_data = laps_data[~laps_data['Compound'].isin(drop_compounds)].reset_index(drop=True)
            else:
                print(f"Warning: No 'Compound' column in {gp}/laps.csv, skipping compound processing")
            all_laps_data.append(laps_data)  # 항상 추가
        else:
            print(f"Warning: Failed to load {gp}/laps.csv")

        weather_data = azure_access.read_csv_by_data_frame(f'{gp}/weather_data.csv')
        if weather_data is not None:
            print(f"Loaded {gp}/weather_data.csv with columns: {weather_data.columns.tolist()}")
            weather_data['Time'] = pd.to_timedelta(weather_data['Time'].astype(str).str.replace('0 days', '').str.strip())
            weather_data['Time'] = base_time + weather_data['Time']
            weather_data['GrandPrix'] = gp
            weather_data = weather_data.sort_values('Time')
            all_weather_data.append(weather_data)
        else:
            print(f"Warning: Failed to load {gp}/weather_data.csv")

    # 데이터 결합
    laps_data = pd.concat(all_laps_data, ignore_index=True)

    # --- 이 코드를 데이터 로딩 및 결합 직후에 추가하세요 ---

    print(f"처리 전 데이터 행 수: {len(laps_data)}")
    print(f"처리 전 'Unknown' 데이터 수: {len(laps_data[laps_data['Compound'] == 'Unknown'])}")

    # 'Compound' 컬럼 값이 'Unknown'이 아닌 행들만 남깁니다.
    laps_data = laps_data[laps_data['Compound'] != 'Unknown'].reset_index(drop=True)

    print(f"처리 후 'Unknown' 데이터 수: {len(laps_data[laps_data['Compound'] == 'Unknown'])}")
    print(f"처리 후 데이터 행 수: {len(laps_data)}")

    weather_data = pd.concat(all_weather_data, ignore_index=True)

    if not weather_data.empty:
        print(f"Loaded weather.csv with columns: {weather_data.columns.tolist()}")

    print("laps_data shape:", laps_data.shape)
    print("laps_data columns:", laps_data.columns)
    print("Sample PitInTime and PitOutTime:\n", laps_data[['LapNumber', 'PitInTime', 'PitOutTime']].head(30))

    # 데이터 전처리 및 정렬 (수정된 부분)
    def fix_time_sorting(df, time_col='Time', group_col='GrandPrix'):
        """
        시간 컬럼의 정렬 문제를 수정하는 함수
        """
        # NaN 값 제거
        df = df.dropna(subset=[time_col]).copy()
        
        # datetime으로 변환
        df[time_col] = pd.to_datetime(df[time_col])
        
        # 각 그룹별로 정렬 확인 및 수정
        fixed_groups = []
        
        for group_name in df[group_col].unique():
            group_data = df[df[group_col] == group_name].copy()
            
            # 시간순으로 정렬
            group_data = group_data.sort_values(time_col).reset_index(drop=True)
            
            # 정렬 확인
            if not group_data[time_col].is_monotonic_increasing:
                print(f"Warning: {group_name} has non-monotonic time values, fixing...")
                # 중복된 시간값이 있을 경우 미세하게 조정
                duplicated = group_data[time_col].duplicated()
                if duplicated.any():
                    print(f"Found {duplicated.sum()} duplicate time values in {group_name}")
                    # 중복된 시간에 microsecond 추가
                    for i, is_dup in enumerate(duplicated):
                        if is_dup:
                            group_data.iloc[i, group_data.columns.get_loc(time_col)] += pd.Timedelta(microseconds=i)
            
            fixed_groups.append(group_data)
            print(f"Fixed {group_name}: {len(group_data)} rows, time range: {group_data[time_col].min()} to {group_data[time_col].max()}")
        
        return pd.concat(fixed_groups, ignore_index=True)

    # 시간 정렬 수정
    print("\n=== Fixing time sorting issues ===")
    laps_data = fix_time_sorting(laps_data)
    weather_data = fix_time_sorting(weather_data)

    # 병합 전 최종 정렬 및 확인
    print("\n=== Final sorting and verification ===")
    laps_data = laps_data.sort_values(['GrandPrix', 'Time']).reset_index(drop=True)
    weather_data = weather_data.sort_values(['GrandPrix', 'Time']).reset_index(drop=True)

    # 정렬 상태 최종 확인
    print("Final sorting verification:")
    for gp in laps_data['GrandPrix'].unique():
        laps_gp = laps_data[laps_data['GrandPrix'] == gp]['Time']
        weather_gp = weather_data[weather_data['GrandPrix'] == gp]['Time'] if gp in weather_data['GrandPrix'].unique() else pd.Series(dtype='datetime64[ns]')
        
        laps_sorted = laps_gp.is_monotonic_increasing
        weather_sorted = weather_gp.is_monotonic_increasing if not weather_gp.empty else True
        
        print(f"{gp}: laps_sorted={laps_sorted}, weather_sorted={weather_sorted}")
        
        if not laps_sorted or not weather_sorted:
            print(f"ERROR: {gp} is not properly sorted!")
            break

    # 시간 차이 계산 (수정)
    print("\n=== Analyzing time differences ===")
    laps_times = laps_data.sort_values(['GrandPrix', 'Time'])['Time']
    weather_times = weather_data.sort_values(['GrandPrix', 'Time'])['Time']
    time_diff = pd.Series(dtype='timedelta64[ns]', index=laps_times.index)
    for idx in laps_times.index:
        nearest_idx = weather_times.searchsorted(laps_times[idx], side='left') - 1
        if nearest_idx >= 0 and nearest_idx < len(weather_times):
            nearest_weather_time = weather_times.iloc[nearest_idx]
            time_diff[idx] = abs(laps_times[idx] - nearest_weather_time)
        else:
            time_diff[idx] = pd.Timedelta(seconds=0)  # 기본값 설정
    print(f"Max time difference: {time_diff.max()}")
    print(f"Mean time difference: {time_diff[time_diff > pd.Timedelta(0)].mean().total_seconds():.2f} seconds")
    print(f"Median time difference: {time_diff[time_diff > pd.Timedelta(0)].median().total_seconds():.2f} seconds")

    # Fixed merge section - replace the existing merge attempt section

    print("\n=== Attempting merge (FIXED) ===")

    # Check if we have weather data to merge
    if not weather_data.empty:
        # Create separate dataframes for each Grand Prix and merge individually
        merged_groups = []
        
        for gp in laps_data['GrandPrix'].unique():
            print(f"Processing {gp}...")
            
            # Get data for this specific Grand Prix
            laps_gp = laps_data[laps_data['GrandPrix'] == gp].copy()
            weather_gp = weather_data[weather_data['GrandPrix'] == gp].copy()
            
            if weather_gp.empty:
                print(f"  No weather data for {gp}, using default values")
                # Add default weather values
                if not weather_data.empty:
                    avg_track_temp = weather_data['TrackTemp'].mean()
                    avg_air_temp = weather_data['AirTemp'].mean() if 'AirTemp' in weather_data.columns else 25.0
                    laps_gp['TrackTemp'] = avg_track_temp
                    if 'AirTemp' in weather_data.columns:
                        laps_gp['AirTemp'] = avg_air_temp
                    if 'Humidity' in weather_data.columns:
                        laps_gp['Humidity'] = weather_data['Humidity'].mean()
                    if 'Pressure' in weather_data.columns:
                        laps_gp['Pressure'] = weather_data['Pressure'].mean()
                    if 'WindDirection' in weather_data.columns:
                        laps_gp['WindDirection'] = weather_data['WindDirection'].mean()
                    if 'WindSpeed' in weather_data.columns:
                        laps_gp['WindSpeed'] = weather_data['WindSpeed'].mean()
                    if 'Rainfall' in weather_data.columns:
                        laps_gp['Rainfall'] = 0
                merged_groups.append(laps_gp)
                continue
            
            # Sort by Time only (critical for merge_asof)
            laps_gp = laps_gp.sort_values('Time').reset_index(drop=True)
            weather_gp = weather_gp.sort_values('Time').reset_index(drop=True)
            
            # Verify sorting
            if not laps_gp['Time'].is_monotonic_increasing:
                print(f"  ERROR: laps data for {gp} is not properly time-sorted")
                continue
            if not weather_gp['Time'].is_monotonic_increasing:
                print(f"  ERROR: weather data for {gp} is not properly time-sorted")
                continue
            
            try:
                # Perform merge_asof for this Grand Prix
                merged_gp = pd.merge_asof(
                    laps_gp,
                    weather_gp.drop(columns=['GrandPrix']),  # Remove GrandPrix to avoid duplication
                    on='Time',
                    direction='nearest',
                    tolerance=pd.Timedelta(minutes=20)  # 20 minute tolerance
                )
                
                print(f"  Successfully merged {len(merged_gp)} rows")
                merged_groups.append(merged_gp)
                
            except Exception as e:
                print(f"  Merge failed for {gp}: {e}")
                # Add default weather values for this GP
                if not weather_data.empty:
                    avg_track_temp = weather_data['TrackTemp'].mean()
                    laps_gp['TrackTemp'] = avg_track_temp
                    if 'AirTemp' in weather_data.columns:
                        laps_gp['AirTemp'] = weather_data['AirTemp'].mean()
                merged_groups.append(laps_gp)
        
        # Combine all merged groups
        if merged_groups:
            laps_data = pd.concat(merged_groups, ignore_index=True)
            print("Merge completed successfully!")
            print(f"Final merged data shape: {laps_data.shape}")
            
            # Check what weather columns were successfully merged
            weather_cols = ['TrackTemp', 'AirTemp', 'Humidity', 'Pressure', 'WindDirection', 'WindSpeed', 'Rainfall']
            available_weather_cols = [col for col in weather_cols if col in laps_data.columns]
            print(f"Available weather columns: {available_weather_cols}")
            
            # Show weather data statistics
            for col in available_weather_cols:
                if laps_data[col].notna().sum() > 0:
                    print(f"  {col}: {laps_data[col].min():.1f} to {laps_data[col].max():.1f} (mean: {laps_data[col].mean():.1f})")
        else:
            print("No data groups were successfully processed")
            
    else:
        print("No weather data available for merging")
        # Add default weather columns
        laps_data['TrackTemp'] = 31.2  # Default track temperature
        laps_data['AirTemp'] = 25.0    # Default air temperature

    print(f"Final data shape after merge: {laps_data.shape}")
    print(f"Columns after merge: {laps_data.columns.tolist()}")

    # 나머지 전처리
    print("\n=== Data preprocessing ===")

    # Boolean 열 처리
    boolean_cols = ['IsPersonalBest', 'FreshTyre', 'Deleted', 'FastF1Generated', 'IsAccurate']
    if 'Rainfall' in laps_data.columns:
        boolean_cols.append('Rainfall')

    for col in boolean_cols:
        if col in laps_data.columns:
            laps_data[col] = pd.to_numeric(laps_data[col], errors='coerce').fillna(0).astype('Int64')

    # Special handling for pit-related columns
    laps_data['IsPitStop'] = laps_data['PitInTime'].notnull().astype('Int64')

    # 'LapTime' 및 섹터 타임 변환
    time_cols = ['LapTime', 'Sector1Time', 'Sector2Time', 'Sector3Time']
    for col in time_cols:
        if col in laps_data.columns and laps_data[col].dtype != 'float64':
            laps_data[col] = laps_data[col].apply(lambda x: pd.to_timedelta(str(x).replace('0 days', '')).total_seconds() if pd.notna(x) else np.nan)

    # Drop rows where 'LapTime' is missing
    laps_data = laps_data.dropna(subset=['LapTime', 'Compound', 'TyreLife'])

    # SpeedI1, SpeedFL 보간
    if 'SpeedI1' in laps_data.columns:
        laps_data['SpeedI1'] = laps_data['SpeedI1'].interpolate(method='linear').fillna(laps_data['SpeedI1'].median())
    if 'SpeedFL' in laps_data.columns:
        laps_data['SpeedFL'] = laps_data['SpeedFL'].interpolate(method='linear').fillna(laps_data['SpeedFL'].median())

    # 피처 엔지니어링: 타이어 마모율
    laps_data['TyreWear'] = laps_data.groupby(['Driver', 'Stint'])['TyreLife'].transform('max') - laps_data['TyreLife']

    # 피처 엔지니어링: LapTime 변화율
    laps_data['LapTime_Delta'] = laps_data.groupby(['Driver', 'Stint'])['LapTime'].diff().fillna(0)
    if 'SpeedI1' in laps_data.columns:
        laps_data['SpeedI1_Delta'] = laps_data.groupby(['Driver', 'Stint'])['SpeedI1'].diff().fillna(0)
    laps_data['TyreWear_Rate'] = laps_data.groupby(['Driver', 'Stint'])['TyreWear'].diff().fillna(0) / laps_data['LapNumber']

    # Define column categories
    numerical_cols = ['LapNumber', 'Stint', 'TyreLife', 'Position', 'LapTime_Delta', 'TyreWear_Rate']
    if 'SpeedI1' in laps_data.columns:
        numerical_cols.extend(['SpeedI1', 'SpeedI1_Delta'])
    if 'SpeedFL' in laps_data.columns:
        numerical_cols.append('SpeedFL')
    if 'TrackTemp' in laps_data.columns:
        numerical_cols.append('TrackTemp')
    if 'AirTemp' in laps_data.columns:
        numerical_cols.append('AirTemp')
    if 'Rainfall' in laps_data.columns:
        numerical_cols.append('Rainfall')

    categorical_cols = ['Compound', 'Driver', 'Team']
    if 'TrackStatus' in laps_data.columns:
        categorical_cols.append('TrackStatus')

    # Boolean 열 처리 (존재하는 컬럼만)
    boolean_cols_candidate = ['IsPersonalBest', 'FreshTyre', 'Deleted', 'FastF1Generated', 'IsAccurate', 'Rainfall']
    boolean_cols = [col for col in boolean_cols_candidate if col in laps_data.columns]

    print(f"Processing boolean columns: {boolean_cols}")
    for col in boolean_cols:
        # NaN을 0으로 대체하고 nullable 정수형으로 변환
        laps_data[col] = pd.to_numeric(laps_data[col], errors='coerce').fillna(0).astype('Int64')

    # Special handling for pit-related columns
    laps_data['IsPitStop'] = laps_data['PitInTime'].notnull().astype('Int64')
    print("Created IsPitStop column")

    # 'LapTime' 및 섹터 타임 변환
    time_cols_candidate = ['LapTime', 'Sector1Time', 'Sector2Time', 'Sector3Time']
    time_cols = [col for col in time_cols_candidate if col in laps_data.columns]

    print(f"Processing time columns: {time_cols}")
    for col in time_cols:
        if laps_data[col].dtype != 'float64':
            print(f"Converting {col} from {laps_data[col].dtype} to seconds")
            laps_data[col] = laps_data[col].apply(
                lambda x: pd.to_timedelta(str(x).replace('0 days', '')).total_seconds() 
                if pd.notna(x) else np.nan
            )

    # Drop rows where 'LapTime' is missing
    if 'LapTime' in laps_data.columns:
        before_drop = len(laps_data)
        laps_data = laps_data.dropna(subset=['LapTime'])
        after_drop = len(laps_data)
        print(f"Dropped {before_drop - after_drop} rows with missing LapTime")

    # SpeedI1, SpeedFL 보간 (존재하는 컬럼만)
    speed_cols = ['SpeedI1', 'SpeedFL', 'SpeedI2', 'SpeedST']
    for col in speed_cols:
        if col in laps_data.columns:
            print(f"Interpolating {col}")
            laps_data[col] = laps_data[col].interpolate(method='linear').fillna(laps_data[col].median())

    # 피처 엔지니어링: 타이어 마모율
    if all(col in laps_data.columns for col in ['Driver', 'Stint', 'TyreLife']):
        laps_data['TyreWear'] = laps_data.groupby(['Driver', 'Stint'])['TyreLife'].transform('max') - laps_data['TyreLife']
        print("Created TyreWear feature")

    # 피처 엔지니어링: LapTime 변화율
    if all(col in laps_data.columns for col in ['Driver', 'Stint', 'LapTime']):
        laps_data['LapTime_Delta'] = laps_data.groupby(['Driver', 'Stint'])['LapTime'].diff().fillna(0)
        print("Created LapTime_Delta feature")

    if all(col in laps_data.columns for col in ['Driver', 'Stint', 'SpeedI1']):
        laps_data['SpeedI1_Delta'] = laps_data.groupby(['Driver', 'Stint'])['SpeedI1'].diff().fillna(0)
        print("Created SpeedI1_Delta feature")

    if all(col in laps_data.columns for col in ['Driver', 'Stint', 'TyreWear', 'LapNumber']):
        laps_data['TyreWear_Rate'] = laps_data.groupby(['Driver', 'Stint'])['TyreWear'].diff().fillna(0) / laps_data['LapNumber']
        print("Created TyreWear_Rate feature")

    # Define column categories (존재하는 컬럼만)
    numerical_cols_candidate = [
        'LapNumber', 'Stint', 'SpeedI1', 'SpeedI2', 'SpeedFL', 'SpeedST', 
        'TyreLife', 'Position', 'TrackTemp', 'AirTemp', 'Humidity', 'Pressure', 'WindDirection', 'WindSpeed',
        'LapTime_Delta', 'SpeedI1_Delta', 'TyreWear_Rate', 'TyreWear', 'Rainfall'
    ]
    numerical_cols = [col for col in numerical_cols_candidate if col in laps_data.columns]

    categorical_cols_candidate = ['Compound', 'TrackStatus', 'Driver', 'Team', 'DriverNumber']
    categorical_cols = [col for col in categorical_cols_candidate if col in laps_data.columns]

    print(f"\nFinal column categories:")
    print(f"Numerical columns ({len(numerical_cols)}): {numerical_cols}")
    print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")

    # 데이터 품질 체크
    print(f"\nData quality check:")
    print(f"Total rows: {len(laps_data)}")
    print(f"Total columns: {len(laps_data.columns)}")

    # Missing values check for key columns
    key_columns = ['LapTime', 'Driver', 'LapNumber', 'GrandPrix', 'Compound', 'TyreLife']
    for col in key_columns:
        if col in laps_data.columns:
            missing_count = laps_data[col].isnull().sum()
            print(f"{col}: {missing_count} missing values ({missing_count/len(laps_data)*100:.2f}%)")

    # PitStop 관련 통계
    if 'IsPitStop' in laps_data.columns:
        pit_stops = laps_data[laps_data['IsPitStop'] == 1]
        print(f"\nPit stop statistics:")
        print(f"Total pit stops: {len(pit_stops)}")
        print(f"Pit stops with PitInTime: {pit_stops['PitInTime'].notna().sum()}")
        print(f"Pit stops with PitOutTime: {pit_stops['PitOutTime'].notna().sum()}")

    # GrandPrix별 데이터 분포
    if 'GrandPrix' in laps_data.columns:
        print(f"\nData distribution by GrandPrix:")
        gp_counts = laps_data['GrandPrix'].value_counts()
        for gp, count in gp_counts.items():
            print(f"  {gp}: {count} rows")


    print("\nData preprocessing completed successfully!")

    print("\nPHASE 2: 데이터셋 분리 시작")
    laps_data_grouped = laps_data.copy()
    laps_data_single = laps_data[laps_data['GrandPrix'].str.contains(track_name, na=False)].copy()

    dry_laps_data_grouped = laps_data_grouped[laps_data_grouped['Rainfall'] == 0].copy()
    wet_laps_data_grouped = laps_data_grouped[laps_data_grouped['Rainfall'] == 1].copy()
    dry_laps_data_single = laps_data_single[laps_data_single['Rainfall'] == 0].copy()

    has_wet_data = not wet_laps_data_grouped.empty
    print("PHASE 2: 완료")