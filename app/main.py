from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from database import SessionLocal, Message, Character, CharacterPrompt  # DB 세션 및 모델 가져오기
from typing import List, Dict  # Dict 추가
import uuid

from openai_api import get_openai_response  # OpenAI API 호출 모듈

app = FastAPI()

# CORS 설정: 모든 도메인, 메서드, 헤더를 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # 모든 도메인 허용
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
    favorability: int
    character_appearance: Dict
    character_personality: Dict
    character_background: Dict
    character_speech_style: Dict
    example_dialogues: List[Dict]
    chat_history: str

@app.post("/generate/")
def generate_response(request: GenerateRequest):
    """
    LangChain을 이용해 사용자 요청 처리 및 캐릭터 응답 생성 API.
    """
    try:
        print("Received request data:", request.dict())  # 디버깅용 로그

        # OpenAI API를 통해 캐릭터 응답 생성
        bot_response = get_openai_response(
            user_message=request.user_message,
            character_name=request.character_name,
            favorability=request.favorability,
            appearance=request.character_appearance,
            personality=request.character_personality,
            background=request.character_background,
            speech_style=request.character_speech_style,
            example_dialogues=request.example_dialogues,
            chat_history=request.chat_history
        )

        print("OpenAI response:", bot_response)  # 디버깅용 로그

        # 응답 반환
        return {
            "text": bot_response.get("response", ""),
            "emotion": bot_response.get("emotion", "Neutral"),
            "favorability": bot_response.get("updated_likes", request.favorability)
        }

    except Exception as e:
        print(f"Error in generate_response: {str(e)}")  # 디버깅용 로그
        raise HTTPException(status_code=500, detail=str(e))

# uvicorn main:app --reload --log-level debug --port 8001 