# f1_optimizer/config.py

# Azure Blob Storage 정보 (이제 data_container 모듈에서 직접 관리되므로 여기서는 필요 없습니다)

# 컴파운드 맵
COMPOUND_MAP = {
    'SOFT': 'SOFT',
    'SUPERSOFT': 'SOFT',
    'ULTRASOFT': 'SOFT',
    'HYPERSOFT': 'SOFT',
    'MEDIUM': 'MEDIUM',
    'HARD': 'HARD',
    'INTERMEDIATE': 'INTERMEDIATE',
    'WET': 'WET',
    'TEST': 'TEST',
    'TEST_UNKNOWN': 'Unknown',
    'UNKNOWN': 'Unknown'
}

# 처리할 트랙 목록
TRACKS_TO_PROCESS = [
    "Abu_Dhabi", "Australian", "Austrian", "Azerbaijan", "Bahrain",
    "Belgian", "British", "Canadian", "Chinese", "Dutch",
    "Eifel", "Emilia_Romagna", "French", "German", "Hungarian",
    "Italian", "Japanese", "Las_Vegas", "Mexico_City", "Miami",
    "Monaco", "Portuguese", "Qatar",
    "Russian", "Sakhir",
    "Saudi_Arabian", "Singapore", "Spanish", "Styrian",
    # "Sao_Paulo", 
    "Turkish", "Tuscan", "United_States"
]

# 평가할 시작 컴파운드 목록
COMPOUNDS_TO_EVALUATE = ["HARD", "MEDIUM", "SOFT"]

# 경로 설정
# sys.path에 추가될 프로젝트의 루트 경로
PROJECT_ROOT_PATH = "/home/azureuser/cloudfiles/code"

# 설정 및 결과물 저장 경로
CONFIG_FILE_PATH = "/home/azureuser/cloudfiles/code/Users/data/tracks_config.json"
ARTIFACTS_PATH = "/home/azureuser/cloudfiles/code/Users/project/src/tyre_strategy_module/json_result"
PARAMS_JSON_PATH = f"{ARTIFACTS_PATH}/complete"
RESULT_JSON_PATH = f"{ARTIFACTS_PATH}/result"