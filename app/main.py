from starlette.websockets import WebSocketState  # Starlette에서 WebSocket 상태 상수 가져오기
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from database import SessionLocal, ChatLog # DB 세션 가져오기
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime
import asyncio
import uuid
import os

from openai_api import get_openai_response  # OpenAI API 호출 모듈

app = FastAPI()

CLIENT_DOMAIN = os.getenv("CLIENT_DOMAIN")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=[CLIENT_DOMAIN],  # 모든 도메인 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
    nickname: dict
    user_unique_name: Optional[str]
    user_introduction: Optional[str]
    favorability: int
    character_appearance: str
    character_personality: str
    character_background: str
    character_speech_style: str
    example_dialogues: List[Any]
    chat_history: str

# 웹소켓 연결 관리
class Chat:
    def __init__(self):
        self.active_connections: Dict[str, Tuple[WebSocket, str]] = {}  # 세션 ID와 웹소켓 연결 및 채팅룸 id 매핑
        self.inactive_tasks: Dict[str, asyncio.Task] = {}  # 세션별 비활성화 타이머 관리
        self.logs_path = "chat_logs"  # 로그 파일 디렉토리

        # chat_log 디렉토리가 없을 경우 새로 생성
        if not os.path.exists(self.logs_path):
            os.makedirs(self.logs_path)

    # 연결설정: 클라이언트 연결 수락하고 세션 id 생성
    async def connect(self, websocket: WebSocket, room_id: str):
        # 기존 세션 ID 확인
        for session_id, (existing_websocket, existing_room_id) in self.active_connections.items():
            if existing_room_id == room_id:
                # 동일한 room_id에 대한 연결이 이미 있으면 해당 세션 ID 반환
                print(f"Reusing existing session ID: {session_id}")
                await websocket.accept()  # 새 연결도 수락
                self.active_connections[session_id] = (websocket, room_id)  # WebSocket을 업데이트
                return session_id

        # 새로운 세션 생성
        await websocket.accept()
        session_id = str(uuid.uuid4())
        self.active_connections[session_id] = (websocket, room_id)

         # 세션 로그 파일 생성
        log_path = f"{self.logs_path}/{session_id}.log"
        with open(log_path, "a", encoding="utf-8") as log_file:
            start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_file.write(f"Session opened at: {start_time}\n")

        # 비활성화 타이머 시작 (10분)
        self.inactive_tasks[session_id] = asyncio.create_task(self.inactivity_check(session_id))
        return session_id

    # 연결 해제 - 로그 파일을 DB에 저장하고 실행중인 타이머가 있다면 취소
    def disconnect(self, session_id: str):
        if session_id not in self.active_connections:
            print(f"Session {session_id} already disconnected.")
            return
        
        websocket, room_id = self.active_connections.pop(session_id, (None, None))
        # print(f"Disconnected session. room_id: {room_id}, websocket: {websocket}")
        log_path = f"{self.logs_path}/{session_id}.log"

        if websocket:
            end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # 로그 파일이 존재하고 실제 대화 내용이 있는지 확인
            if os.path.exists(log_path):
                with open(log_path, "r", encoding="utf-8") as log_file:
                    log_lines = log_file.readlines()
                    has_content = any("user:" in line or "chatbot:" in line for line in log_lines)

                # 대화 내용이 없는 경우 로그 파일 삭제
                if not has_content:
                    os.remove(log_path)
                else:
                    # 대화 내용이 있는 경우에만 종료 시간 기록 후 DB에 저장
                    with open(log_path, "a", encoding="utf-8") as log_file:
                        log_file.write(f"Session closed at: {end_time}\n")

                    # 로그 DB 저장
                    try:
                        with SessionLocal() as db:  # 데이터베이스 세션 생성
                            self.save_log_to_db(session_id, room_id, end_time, db)
                            os.remove(log_path)
                    except Exception as e:
                        print(f"Error saving log to database: {e}")

        # 비활성화 타이머가 있는 경우 취소
        if session_id in self.inactive_tasks:
            self.inactive_tasks[session_id].cancel()
            del self.inactive_tasks[session_id]

    async def log_message(self, session_id: str, sender: str, message: str):
        log_path = f"{self.logs_path}/{session_id}.log"
        with open(log_path, "a", encoding="utf-8") as log_file:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_file.write(f"[{timestamp}] {sender}: {message}\n")
            log_file.flush()

    async def send_message(self, websocket: WebSocket, session_id: str, message: dict):
        """
        WebSocket으로 메시지를 전송합니다.
        """
        try:
            if websocket.application_state != WebSocketState.CONNECTED:
                print(f"WebSocket is not connected for session {session_id}.")
                return
            await websocket.send_json(message)  # JSON 데이터를 전송

            # 응답 로그 기록
            if "text" in message:
                await self.log_message(session_id, "chatbot", message["text"])
        except WebSocketDisconnect:
            print(f"WebSocket disconnected for session while sending message {session_id}.")
        except Exception as e:
            print(f"Error sending message for session {session_id}: {str(e)}")

    # 자리비움 타이머 설정 및 일정 시간 초과시 세션 연결 해제
    async def inactivity_check(self, session_id: str):
        try:
            while True:
                await asyncio.sleep(600)  # 10분(600초) 대기
                websocket, _ = self.active_connections.get(session_id, (None, None))
                if websocket and websocket.application_state == WebSocketState.CONNECTED:
                    await websocket.send_json({
                        "sender": "bot",
                        "message": "10분 동안 활동이 없어 연결이 종료됩니다."
                    })
                    await websocket.close()
                self.disconnect(session_id)
        except asyncio.CancelledError:
            # 타이머가 취소된 경우 무시
            pass
        except Exception as e:
            print(f"Error in inactivity_check for session {session_id}: {str(e)}")

    # 데이터베이스에 로그 저장
    def save_log_to_db(self, session_id: str, room_id: str, end_time: str, db: Session = Depends(get_db)):
        log_path = f"{self.logs_path}/{session_id}.log"
        with open(log_path, "r", encoding="utf-8") as log_file:
            log_content = log_file.read()

        # 로그파일에서 세션시작시간 추출
        try:
            start_time = next(
                line.split(": ")[-1] for line in log_content.splitlines()
                if "Session opened at:" in line
            )
            start_time = datetime.strptime(start_time.strip(), "%Y-%m-%d %H:%M:%S")  # 문자열 -> datetime 변환
        except StopIteration:
            raise ValueError("로그 파일에 'Session opened at:'이 없습니다.")
        
        # ChatLog ORM 인스턴스 생성
        chat_log = ChatLog(
            session_id=session_id,
            chat_id=room_id,
            start_time=start_time,
            end_time=end_time,
            log=log_content
        )

        # 데이터베이스에 저장
        db.add(chat_log)
        db.commit()


chat = Chat()


@app.websocket("/ws/generate/")
async def websocket_generate(websocket: WebSocket, room_id: str):
    """
    LangChain을 이용해 사용자 요청 처리 및 캐릭터 응답 생성 API (웹소켓).
    """
    session_id = await chat.connect(websocket, room_id)
    try:
        while True:
            # 클라이언트에서 요청 데이터를 수신
            try:
                data = await websocket.receive_json()
                request = GenerateRequest(**data)
            except ValueError as e:
                print(f"Invalid JSON format: {e}")
                await websocket.close(code=1003, reason="Invalid JSON format")
                return
            except WebSocketDisconnect as e:
                print(f"WebSocket disconnected for session {session_id}. Reason: {e.code}")
                break
            except Exception as e:
                print(f"Error parsing request: {e}")
                await chat.send_message(websocket, session_id, {"error": "Invalid data format"})
                continue

            # print("Received request data:", request.dict())  # 디버깅용 로그

            # 사용자 메시지 로그 기록
            await chat.log_message(session_id, "user", request.user_message)

            try:
                # OpenAI API를 통해 캐릭터 응답 생성
                bot_response = get_openai_response(
                    user_message=request.user_message,
                    character_name=request.character_name,
                    nickname=request.nickname,
                    user_unique_name=request.user_unique_name,
                    user_introduction=request.user_introduction,
                    favorability=request.favorability,
                    appearance=request.character_appearance,
                    personality=request.character_personality,
                    background=request.character_background,
                    speech_style=request.character_speech_style,
                    example_dialogues=request.example_dialogues,
                    chat_history=request.chat_history,
                    room_id=room_id
                )

                print("OpenAI response:", bot_response)  # 디버깅용 로그

                # 클라이언트로 응답 전송
                response = {
                    "text": bot_response.get("response", ""),
                    "emotion": bot_response.get("emotion", "Neutral"),
                    "favorability": bot_response.get("favorability", request.favorability)
                }
                await chat.send_message(websocket, session_id, response)
            except Exception as e:
                print(f"Error in websocket_generate: {str(e)}")
                await chat.send_message(websocket, session_id, {"error": str(e)})

    except WebSocketDisconnect as e:
        print(f"WebSocket disconnected unexpectedly for session {session_id}. Reason: {e.code}")
    except Exception as e:
        print(f"Unexpected error in WebSocket handling for session {session_id}: {str(e)}")
    finally:
        chat.disconnect(session_id)
        print(f"Session {session_id} disconnected.")

@app.get("/")
async def root():
    return {"message": "Welcome to the Hell..ow World"}

# uvicorn main:app --reload --log-level debug --port 8001
