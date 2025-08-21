# ==============================================================================
# F1 Race Strategy Database Manager
#
# 설명:
#   이 스크립트는 F1 레이스 전략이 담긴 JSON 파일을 Oracle 데이터베이스에
#   저장하거나, 데이터베이스에 저장된 전략을 조회하는 두 가지 기능을 제공합니다.
#
# 사전 준비:
#   - oracledb 라이브러리가 설치되어 있어야 합니다.
#     (pip install oracledb)
#
# 실행 방법:
#   1. 모든 JSON 파일 DB에 저장하기:
#      python strategy_db_manager.py insert /path/to/json/folder
#
#   2. 특정 전략 DB에서 조회하기:
#      python strategy_db_manager.py get [track_name] [starting_compound]
#      (예: python strategy_db_manager.py get Sakhir SOFT)
#
# ==============================================================================

import oracledb
import json
import os
import sys

# --- 데이터베이스 연결 정보 ---
# 환경에 맞게 수정하여 사용하세요.
DB_URL = "track/track@9.234.242.124:1521/xepdb1"


class DBManager:
    """Oracle DB 연결 및 종료를 관리하는 헬퍼 클래스"""
    @staticmethod
    def makeConCur(url):
        """데이터베이스에 연결하고 커넥션과 커서 객체를 반환합니다."""
        con = oracledb.connect(url)
        cur = con.cursor()
        return con, cur

    @staticmethod
    def closeConCur(con, cur):
        """커서와 커넥션을 순서대로 닫습니다."""
        if cur:
            cur.close()
        if con:
            con.close()


def insert_strategy_from_file(file_path, db_url):
    """JSON 파일을 읽어 Oracle DB에 레이스 전략 정보를 저장합니다."""
    
    if not os.path.exists(file_path):
        print(f"경고: 파일을 찾을 수 없습니다. 건너뜁니다: {file_path}")
        return

    con, cur = None, None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        con, cur = DBManager.makeConCur(db_url)

        # 1. RACE_STRATEGIES 테이블에 마스터 데이터 삽입
        sql_strategy = """
            INSERT INTO RACE_STRATEGIES (
                track_name, starting_compound, total_laps, 
                estimated_race_time, total_pit_stops, strategy_summary
            ) VALUES (
                :track_name, :starting_compound, :total_laps, 
                :estimated_race_time, :total_pit_stops, :strategy_summary
            ) RETURNING strategy_id INTO :strategy_id
        """
        
        new_strategy_id = cur.var(oracledb.DB_TYPE_NUMBER)
        start_compound = data['stints'][0]['compound'] if data.get('stints') else data.get('initial_compound')

        cur.execute(sql_strategy, 
            track_name=data['track_name'],
            starting_compound=start_compound,
            total_laps=data['total_laps'],
            estimated_race_time=data['estimated_race_time_seconds'],
            total_pit_stops=data['total_pit_stops'],
            strategy_summary=data['strategy_summary'],
            strategy_id=new_strategy_id
        )
        
        inserted_id = new_strategy_id.getvalue()[0]
        print(f"'{data['track_name']}' ({start_compound}) 전략 데이터 삽입 성공. Strategy ID: {inserted_id}")

        # 2. STRATEGY_STINTS 테이블에 스틴트 상세 데이터 반복 삽입
        sql_stint = """
            INSERT INTO STRATEGY_STINTS (
                strategy_id, stint_number, compound, 
                start_lap, end_lap, stint_length
            ) VALUES (
                :strategy_id, :stint_number, :compound, 
                :start_lap, :end_lap, :stint_length
            )
        """
        
        stints_data = [
            {
                "strategy_id": inserted_id,
                "stint_number": stint['stint_number'],
                "compound": stint['compound'],
                "start_lap": stint['start_lap'],
                "end_lap": stint['end_lap'],
                "stint_length": stint['stint_length']
            } for stint in data['stints']
        ]
        
        if stints_data:
            cur.executemany(sql_stint, stints_data)
            print(f"-> {len(stints_data)}개의 스틴트 데이터 삽입 완료.")

        con.commit()

    except Exception as e:
        print(f"오류가 발생했습니다: {e}")
        if con:
            con.rollback()
    finally:
        if con and cur:
            DBManager.closeConCur(con, cur)


def get_strategy_as_json(track_name, starting_compound, db_url):
    """트랙 이름과 시작 컴파운드를 받아 레이스 전략을 조회하고, 결과를 JSON 형식으로 반환합니다."""
    
    con, cur = None, None
    try:
        con, cur = DBManager.makeConCur(db_url)
        
        sql = """
            SELECT
                rs.total_laps, rs.estimated_race_time, rs.total_pit_stops,
                rs.strategy_summary, ss.stint_number, ss.compound,
                ss.start_lap, ss.end_lap, ss.stint_length
            FROM
                RACE_STRATEGIES rs
            JOIN
                STRATEGY_STINTS ss ON rs.strategy_id = ss.strategy_id
            WHERE
                rs.track_name = :t_name 
                AND rs.starting_compound = :s_compound
            ORDER BY
                ss.stint_number
        """
        
        cur.execute(sql, t_name=track_name, s_compound=starting_compound)
        results = cur.fetchall()
        
        if not results:
            return None

        first_row = results[0]
        strategy_dict = {
            "track_name": track_name,
            "starting_compound": starting_compound,
            "total_laps": first_row[0],
            "estimated_race_time_seconds": first_row[1],
            "estimated_race_time_minutes": round(first_row[1] / 60, 2),
            "total_pit_stops": first_row[2],
            "strategy_summary": first_row[3],
            "stints": [
                {
                    "stint_number": row[4],
                    "compound": row[5],
                    "start_lap": row[6],
                    "end_lap": row[7],
                    "stint_length": row[8]
                } for row in results
            ]
        }
        
        return strategy_dict

    except Exception as e:
        print(f"오류가 발생했습니다: {e}")
        return None
    finally:
        if con and cur:
            DBManager.closeConCur(con, cur)
            print("DB 연결을 종료합니다.")


def get_all_track_names(db_url: str) -> list | None:
    """데이터베이스에서 전략이 저장된 모든 트랙의 고유한 목록을 조회하여 리스트로 반환합니다."""
    con, cur = None, None
    try:
        con, cur = DBManager.makeConCur(db_url)
        
        # RACE_STRATEGIES 테이블에서 중복 없이 트랙 이름만 조회하고 알파벳순으로 정렬합니다.
        sql = "SELECT DISTINCT track_name FROM RACE_STRATEGIES ORDER BY track_name ASC"
        
        cur.execute(sql)
        
        # cur.fetchall()의 결과는 [('Abu_Dhabi',), ('Australian',)] 같은 튜플의 리스트입니다.
        # 이를 ['Abu_Dhabi', 'Australian'] 형태의 문자열 리스트로 변환합니다.
        results = [row[0] for row in cur.fetchall()]
        
        return results

    except Exception as e:
        print(f"DB에서 트랙 목록 조회 중 오류가 발생했습니다: {e}")
        return None # 오류 발생 시 None 반환
    finally:
        if con and cur:
            DBManager.closeConCur(con, cur)


def run_bulk_insertion(json_folder_path):
    """지정된 폴더의 모든 전략 JSON 파일을 DB에 저장합니다."""
    tracks = [
        "Abu_Dhabi", "Australian", "Austrian", "Azerbaijan", "Bahrain",
        "Belgian", "British", "Canadian", "Chinese", "Dutch", "Eifel",
        "Emilia_Romagna", "French", "German", "Hungarian", "Italian",
        "Japanese", "Las_Vegas", "Mexico_City", "Miami", "Monaco",
        "Portuguese", "Qatar", "Russian", "Sakhir", "Saudi_Arabian",
        "Singapore", "Spanish", "Styrian", "Sao_Paulo", "Turkish",
        "Tuscan", "United_States"
    ]
    compounds = ["HARD", "MEDIUM", "SOFT"]

    print("--- 전체 전략 데이터베이스 저장을 시작합니다 ---")
    for tname in tracks:
        for cname in compounds:
            # 파일 이름 형식이 노트북과 약간 다른 것을 감안하여 수정
            file_name = f"{tname}_{cname}_strategy.json" 
            file_path = os.path.join(json_folder_path, file_name)
            
            print(f"\n-> {file_name} 파일 처리 시작...")
            insert_strategy_from_file(file_path, DB_URL)
    print("\n--- 모든 파일 처리가 완료되었습니다 ---")


def print_usage():
    """스크립트 사용법을 출력합니다."""
    print("사용법:")
    print("  - 모든 JSON 파일 저장: python strategy_db_manager.py insert <json_folder_path>")
    print("    (예: python scripts/strategy_db_manager.py insert /path/to/json/result)")
    print("  - 특정 전략 조회: python strategy_db_manager.py get <track_name> <compound>")
    print("    (예: python scripts/strategy_db_manager.py get Sakhir SOFT)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)

    mode = sys.argv[1].lower()

    if mode == 'insert':
        if len(sys.argv) != 3:
            print("오류: JSON 파일이 있는 폴더 경로를 입력해야 합니다.")
            print_usage()
            sys.exit(1)
        json_folder = sys.argv[2]
        run_bulk_insertion(json_folder)
    
    elif mode == 'get':
        if len(sys.argv) != 4:
            print("오류: 트랙 이름과 시작 컴파운드를 모두 입력해야 합니다.")
            print_usage()
            sys.exit(1)
        
        track_name = sys.argv[2]
        start_compound = sys.argv[3].upper()
        
        strategy_result = get_strategy_as_json(track_name, start_compound, DB_URL)
        
        if strategy_result:
            json_output = json.dumps(strategy_result, indent=2, ensure_ascii=False)
            print(json_output)
        else:
            print(f"'{track_name}' 트랙의 '{start_compound}' 시작 전략을 찾을 수 없습니다.")
    
    else:
        print(f"오류: 알 수 없는 모드 '{mode}' 입니다.")
        print_usage()