# scripts/repeat.py
import sys
import os
import time

# 프로젝트 루트 경로를 sys.path에 추가하여 f1_optimizer 모듈을 찾을 수 있도록 함
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# config 파일에서 트랙 및 컴파운드 목록을 가져옴
from f1_optimizer import config
# predict.py 파일의 main 함수를 가져옴 (이름 충돌을 피하기 위해 별칭(alias) 사용)
from scripts.predict import main as run_single_prediction

def run_all_predictions():
    """
    설정 파일에 정의된 모든 트랙과 컴파운드에 대해 예측을 실행합니다.
    """
    start_time = time.time()
    
    tracks_to_run = config.TRACKS_TO_PROCESS
    compounds_to_run = config.COMPOUNDS_TO_EVALUATE
    
    total_tasks = len(tracks_to_run) * len(compounds_to_run)
    completed_tasks = 0
    failed_tasks = []

    print(f"--- 총 {total_tasks}개의 전략 예측 작업을 시작합니다. ---")

    # 모든 트랙과 컴파운드 조합에 대해 반복 실행
    for track in tracks_to_run:
        for compound in compounds_to_run:
            completed_tasks += 1
            print(f"\n[{completed_tasks}/{total_tasks}] ==> Track: {track}, Compound: {compound} 예측 실행...")
            
            try:
                # predict.py의 main 함수를 직접 호출
                run_single_prediction(track, compound)
                print(f"    ✅ [{track} - {compound}] 전략 생성 성공")

            except FileNotFoundError as e:
                # 학습 결과물이 없는 경우, 에러를 기록하고 계속 진행
                error_message = f"    ❌ [{track} - {compound}] 실패: 필요한 파일을 찾을 수 없습니다. ({e.filename})"
                print(error_message)
                failed_tasks.append(f"{track} - {compound}: File Not Found")

            except Exception as e:
                # 그 외 예기치 못한 에러 발생 시, 기록하고 계속 진행
                error_message = f"    ❌ [{track} - {compound}] 실패: 예측 중 에러 발생"
                print(error_message)
                print(f"    ERROR: {e}")
                failed_tasks.append(f"{track} - {compound}: {e}")

    end_time = time.time()
    print("\n--- 모든 예측 작업을 완료했습니다. ---")
    print(f"총 실행 시간: {end_time - start_time:.2f}초")
    print(f"성공: {total_tasks - len(failed_tasks)}건 / 실패: {len(failed_tasks)}건")

    if failed_tasks:
        print("\n[실패 목록]")
        for task in failed_tasks:
            print(f"- {task}")

if __name__ == "__main__":
    run_all_predictions()