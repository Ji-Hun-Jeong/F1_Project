# scripts/inspect_track_data.py
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 프로젝트 루트 경로를 sys.path에 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
# Users 모듈을 찾기 위한 경로 추가
sys.path.append("/home/azureuser/cloudfiles/code")

from f1_optimizer import data_loader, preprocessing

def analyze_stint_distribution(df, track_name):
    """
    특정 트랙의 타이어 컴파운드별 스틴트 길이 분포를 분석하고 시각화합니다.
    """
    print("\n" + "="*20 + f" {track_name} Stint Length Distribution " + "="*20)
    
    track_df = df[df['GrandPrix'].str.contains(track_name, na=False)].copy()
    
    # PitInTime이 있는 데이터로부터 StintLength 계산
    pit_data = track_df.dropna(subset=['PitInTime', 'LapNumber', 'Driver', 'Compound', 'Time'])
    pit_data = pit_data.sort_values(['GrandPrix', 'Driver', 'Time']) # GrandPrix 기준으로도 정렬
    pit_data['NextPitLap'] = pit_data.groupby(['GrandPrix', 'Driver'])['LapNumber'].shift(-1)
    pit_data['StintLength'] = pit_data['NextPitLap'] - pit_data['LapNumber']
    stints = pit_data.dropna(subset=['StintLength'])
    stints = stints[stints['StintLength'] > 0] # 양수인 스틴트만 사용


    # 컴파운드별 통계 출력
    for compound in ['SOFT', 'MEDIUM', 'HARD']:
        compound_stints = stints[stints['Compound'] == compound]['StintLength']
        if not compound_stints.empty:
            print(f"\n--- {compound} Tyre Stints ---")
            print(compound_stints.describe())
        else:
            print(f"\n--- No data for {compound} Tyre Stints ---")

    # Boxplot으로 시각화
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=stints, x='Compound', y='StintLength', order=['SOFT', 'MEDIUM', 'HARD'])
    plt.title(f'{track_name} - Stint Length Distribution by Compound')
    plt.ylabel('Stint Length (Laps)')
    plt.xlabel('Tyre Compound')
    plt.grid(True)
    # plt.show() # 로컬 환경이 아닐 경우, 파일로 저장
    plt.savefig(f"{track_name}_stint_distribution.png")
    print(f"\n✅ Stint 분포 그래프가 '{track_name}_stint_distribution.png' 파일로 저장되었습니다.")


def analyze_degradation(df, track_name):
    """
    특정 트랙의 타이어 수명에 따른 랩타임 변화(성능 저하)를 분석하고 시각화합니다.
    """
    print("\n" + "="*20 + f" {track_name} Tyre Degradation Analysis " + "="*20)

    track_df = df[df['GrandPrix'].str.contains(track_name, na=False)].copy()
    dry_df = track_df[track_df.get('Rainfall', 0) == 0]
    
    # 랩타임 이상치 제거 (상위 1%, 하위 1% 제거)
    q_low = dry_df["LapTime"].quantile(0.01)
    q_hi  = dry_df["LapTime"].quantile(0.99)
    sane_df = dry_df[(dry_df["LapTime"] < q_hi) & (dry_df["LapTime"] > q_low)]

    # lmplot으로 타이어 수명(TyreLife)과 랩타임(LapTime)의 관계 시각화
    plot = sns.lmplot(data=sane_df, x='TyreLife', y='LapTime', hue='Compound', 
                      hue_order=['SOFT', 'MEDIUM', 'HARD'], height=7, aspect=1.5,
                      scatter_kws={'alpha':0.3})
    plot.ax.grid(True)
    plot.fig.suptitle(f'{track_name} - Tyre Degradation (LapTime vs. TyreLife)', y=1.02)
    # plt.show() # 로컬 환경이 아닐 경우, 파일로 저장
    plot.savefig(f"{track_name}_degradation_analysis.png")
    print(f"\n✅ 성능 저하 분석 그래프가 '{track_name}_degradation_analysis.png' 파일로 저장되었습니다.")


def main(track_name):
    print(f"--- Starting data inspection for {track_name} ---")
    
    # 1. 데이터 로딩 및 전처리 (train.py와 동일한 로직 사용)
    laps_df, weather_df, _ = data_loader.load_and_merge_data(track_name)
    laps_df = preprocessing.fix_time_sorting(laps_df)
    if not weather_df.empty:
        weather_df = preprocessing.fix_time_sorting(weather_df)
    merged_df = preprocessing.merge_weather_data(laps_df, weather_df)
    processed_df = preprocessing.feature_engineer(merged_df)

    # 2. 분석 함수 호출
    analyze_stint_distribution(processed_df, track_name)
    analyze_degradation(processed_df, track_name)

    print("\n--- Data inspection finished ---")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("사용법: python scripts/inspect_track_data.py [track_name]")
        print("예시: python scripts/inspect_track_data.py Monaco")
        sys.exit(1)
    
    track = sys.argv[1]
    main(track)