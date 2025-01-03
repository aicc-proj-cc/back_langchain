from sqlalchemy import create_engine, Column, String, Text, DateTime, ForeignKey, Integer, Boolean, ARRAY, JSON, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, declarative_base
from datetime import datetime
from dotenv import load_dotenv
import os

load_dotenv()

# PostgreSQL 연결 URL
DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# 캐릭터 모델
class Character(Base):
    __tablename__ = "characters"

    char_idx = Column(Integer, primary_key=True, autoincrement=True)  # 캐릭터 인덱스
    user_idx = Column(String(50), nullable=False)  # 회원 인덱스 (임시로 String 사용)
    field_idx = Column(String(50), nullable=False)  # 필드 인덱스 (임시로 String 사용)
    voice_idx = Column(String(50), nullable=False)  # 목소리 인덱스 (임시로 String 사용)
    char_name = Column(String(255), nullable=False)  # 캐릭터 이름
    char_description = Column(Text, nullable=False)  # 캐릭터 한 줄 소개
    character_status_message = Column(ARRAY(Text), nullable=True)  # 캐릭터 상태 메시지 (리스트 형식)
    created_at = Column(DateTime, default=datetime.utcnow)  # 캐릭터 생성 일자
    follows = Column(Integer, default=0)  # 친구 숫자
    is_active = Column(Boolean, default=True)  # 활성화 여부
    favorability = Column(Integer, default=0)  # 호감도
    current_prompt = Column(Text)  # 현재 프롬프트

# 캐릭터 프롬프트 모델
class CharacterPrompt(Base):
    __tablename__ = "char_prompts"

    char_prompt_id = Column(Integer, primary_key=True, autoincrement=True)  # 캐릭터 프롬프트 ID
    char_idx = Column(Integer, ForeignKey("characters.char_idx"), nullable=False)  # 캐릭터 인덱스
    created_at = Column(DateTime, default=datetime.utcnow)  # 생성일자
    character_appearance = Column(JSON, nullable=False)  # 외모 (JSON 형식으로 저장)
    character_personality = Column(JSON, nullable=False)  # 성격 (JSON 형식으로 저장)
    character_background = Column(JSON, nullable=False)  # 배경 (JSON 형식으로 저장)
    character_speech_style = Column(JSON, nullable=False)  # 말투 (JSON 형식으로 저장)
    example_dialogues = Column(ARRAY(JSON), nullable=True)  # 예시 대화 (JSON 형식으로 저장)

# 채팅방 모델
class ChatRoom(Base):
    __tablename__ = "chat_rooms"

    id = Column(String, primary_key=True) # 채팅방 고유 ID
    character_prompt = Column(Text, nullable=False) # 캐릭터 프롬프트
    character_id = Column(Integer, ForeignKey("characters.char_idx")) # 캐릭터 고유 ID (characters db 참조)
    character_name = Column(String(255), nullable=False) # 캐릭터 이름
    character_image = Column(Text, nullable=False) # 캐릭터 이미지
    character_status_message = Column(ARRAY(String), nullable=False)  # 캐릭터 상태 메시지 (리스트 형식)
    character_likes = Column(Integer, nullable=False)  # 캐릭터 호감도
    character_emotion = Column(String, default="보통")  # 캐릭터 기분
    created_at = Column(DateTime, default=datetime.utcnow) # 캐릭터 생성 일자

# 메시지 모델
class Message(Base):
    __tablename__ = "messages"

    id = Column(String, primary_key=True) # 메시지 고유 ID
    room_id = Column(String, ForeignKey("chat_rooms.id")) # 채팅방 ID (chat_rooms db 참조)
    sender = Column(String) # 송신자 이름
    content = Column(Text) # 메시지 내용
    timestamp = Column(DateTime, default=datetime.utcnow) # 메시지 전송 시각

# ChatLogs 테이블
class ChatLog(Base):
    __tablename__ = "chat_logs"

    session_id = Column(String(50), primary_key=True)
    chat_id = Column(String(50), ForeignKey("chat_rooms.id"), nullable=False)
    log = Column(Text, nullable=False)
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, server_default=text("CURRENT_TIMESTAMP"), nullable=False)

# 테이블 생성
Base.metadata.create_all(bind=engine)
