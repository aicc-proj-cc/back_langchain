from fastapi import FastAPI
from pydantic import BaseModel

from openai_api import get_openai_response  # OpenAI API / 캐릭터 챗봇 응답 반환
from fastapi.middleware.cors import CORSMiddleware # CORS 설정용 미들웨어
from database import SessionLocal, Message # DB 세션과 모델 가져오기
import uuid # 고유 ID 생성을 위한 UUID 라이브러리

app = FastAPI()

# CORS 설정: 모든 도메인, 메서드, 헤더를 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 도메인 허용
    allow_credentials=True, # 자격 증명 허용 (쿠키 등)
    allow_methods=["*"], # 모든 HTTP 메서드 허용 (GET, POST 등)
    allow_headers=["*"], # 모든 HTTP 헤더 허용
)


# DB 세션 관리
def get_db():
    """
    데이터베이스 세션을 생성하고 반환.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class GenerateRequest(BaseModel):
    user_message: str
    character_name: str
    prompt: str
    character_likes: int

@app.post("/generate/")
def generate_response(request: GenerateRequest):
    """
    LangChain을 이용해 사용자 요청 처리.
    """
    # json={
    #         "user_message": message,
    #         "room_id": room_id
    #     }

    # print("request.user_message", request.user_message)
    # print("request.character_likes", request.character_likes)


    # OpenAI API를 통해 캐릭터의 응답 생성
    bot_response = get_openai_response(
        user_message=request.user_message,
        prompt=request.prompt,
        character_name=request.character_name,
        character_likes=request.character_likes
        ) # 캐릭터 응답 생성 - 상단의 get_openai_response 라이브러리 참조
    
    return bot_response # 캐릭터 챗봇 응답 반환


# uvicorn main:app --reload --log-level debug --port 8001 