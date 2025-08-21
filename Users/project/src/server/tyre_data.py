# src/server/tyre_data.py

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from Users.project.src.tyre_strategy_module.scripts.strategy_db_manager import get_strategy_as_json, get_all_track_names


# --- 설정 및 전역 변수 ---
DB_URL = "track/track@9.234.242.124:1521/xepdb1"

# --- API 라우터 생성 ---
# 이 tyre_router 객체를 main.py에서 가져가서 사용하게 됩니다.
tyre_router = APIRouter()


# --- API 요청/응답 모델 ---

class StrategyRequest(BaseModel):
    """'/get_strategy' 요청 시 받을 JSON 본문의 형식을 정의합니다."""
    track_name: str
    starting_compound: str


# --- API 엔드포인트 정의 ---

@tyre_router.post("/get_strategy")
async def get_strategy(request: StrategyRequest):
    """
    요청 본문에서 트랙 이름과 시작 컴파운드를 받아
    DB에서 최적의 레이스 전략을 조회하여 반환합니다.
    """
    try:
        # 분리된 모듈의 함수를 호출하여 DB 로직을 실행
        strategy_data = get_strategy_as_json(
            track_name=request.track_name,
            starting_compound=request.starting_compound.upper(),
            db_url=DB_URL
        )

        if strategy_data:
            return JSONResponse(
                status_code=200,
                content={"success": True, "result": strategy_data}
            )
        else:
            return JSONResponse(
                status_code=404,
                content={
                    "success": False,
                    "message": f"'{request.track_name}' 트랙의 '{request.starting_compound}' 시작 전략을 찾을 수 없습니다."
                }
            )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": f"서버 오류가 발생했습니다: {e}"}
        )

class TrackRequest(BaseModel):
    """'/compare_by_track' 요청 시 받을 JSON 본문의 형식을 정의합니다."""
    track_name: str

@tyre_router.post("/compare_by_track")
async def compare_strategies_by_track(request: TrackRequest):
    """
    특정 트랙에 대한 모든 시작 컴파운드(SOFT, MEDIUM, HARD) 전략을
    한 번에 조회하고, 가장 빠른 전략에 'is_recommended' 플래그를 추가하여 반환합니다.
    """
    try:
        track_name = request.track_name
        compounds_to_check = ["SOFT", "MEDIUM", "HARD"]
        strategies = []
        
        # 각 컴파운드에 대해 DB 조회를 반복 실행
        for compound in compounds_to_check:
            strategy_data = get_strategy_as_json(
                track_name=track_name,
                starting_compound=compound,
                db_url=DB_URL
            )
            if strategy_data:
                strategy_data['is_recommended'] = False
                strategies.append(strategy_data)

        if not strategies:
            return JSONResponse(
                status_code=404,
                content={
                    "success": False,
                    "message": f"'{track_name}' 트랙에 대한 전략을 찾을 수 없습니다."
                }
            )
            
        # 가장 빠른 전략 찾기
        fastest_strategy = min(strategies, key=lambda x: x['estimated_race_time_seconds'])
        
        if fastest_strategy['estimated_race_time_seconds'] < 999999.0:
            fastest_strategy['is_recommended'] = True

        return JSONResponse(
            status_code=200,
            content={"success": True, "result": strategies}
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": f"서버 오류가 발생했습니다: {e}"}
        )




@tyre_router.post("/tracks")
async def get_available_tracks():
    """
    데이터베이스에 저장된 모든 F1 트랙의 목록을 반환합니다.
    """
    try:
        # DB 모듈에 새로 추가한 함수를 호출합니다.
        track_list = get_all_track_names(db_url=DB_URL)

        if track_list is not None:
            # 성공적으로 목록을 가져온 경우
            return JSONResponse(
                status_code=200,
                content={"success": True, "result": track_list}
            )
        else:
            # DB 조회 중 오류가 발생하여 None이 반환된 경우
            raise Exception("데이터베이스에서 트랙 목록을 가져오는 데 실패했습니다.")

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": f"서버 오류가 발생했습니다: {e}"}
        )