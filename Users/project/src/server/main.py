from fastapi import FastAPI, WebSocket
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from .response_data import router
from .tyre_data import tyre_router

# 루트 폴더로 이동후
# uvicorn Users.project.src.server.main:app --reload --port=8096 --host=0.0.0.0
app = FastAPI()

# 🔧 CORS 미들웨어 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 개발 중엔 *로 열어두고, 배포 시엔 도메인 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(router)
app.include_router(tyre_router)

@app.get("/")
def gd():
    return {"message": "Hi"}