from fastapi import FastAPI, WebSocket
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from response_data import router  # routes.py에서 router 가져오기

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

@app.get("/")
def gd():
    return {"message": "Hi"}