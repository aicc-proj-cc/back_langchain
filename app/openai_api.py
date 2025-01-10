import json
from collections import Counter
from chat_summary import ChatSummaryMemory
from datetime import datetime
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
import openai
import os
from dotenv import load_dotenv
import logging

# 환경 변수 불러오기
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# LangChain을 위한 OpenAI LLM 인스턴스 생성
llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=openai.api_key)

# 로깅 설정
logging.basicConfig(level=logging.INFO)

# 대화방마다 고유한 대화 이력 관리
class ConversationManager:
    def __init__(self):
        self.conversations = {}

    def get_conversation_memory(self, room_id):
        if room_id not in self.conversations:
            self.conversations[room_id] = ChatSummaryMemory(max_history_length=5)
        return self.conversations[room_id]

# ConversationManager 인스턴스 생성
conversation_manager = ConversationManager()

# 사용자 호칭 설정 함수
def get_user_title(favorability, nickname, user_unique_name):
    try:
        # user_unique_name이 None이거나 빈 문자열이면 nickname을 기본값으로 사용
        user_name = user_unique_name if user_unique_name else nickname

        if favorability < 30:
            return f"{user_name}"
        elif favorability < 70:
            return f"{user_name}"
        else:
            return f"{user_name}"
    except Exception as e:
        logging.error(f"Error in get_user_title: {e}")
        return f"{nickname}"

# 감정 예측 함수
def predict_emotion(user_message):
    try:
        emotion_prompt_template = """
        Analyze the user's message and predict the emotional response of the character.
        Possible emotions are: Happy, Sad, Angry, Confused, Grateful, Embarrassed, or Nervous.

        User Message: {user_message}

        Provide only the predicted emotion.
        """
        emotion_prompt = PromptTemplate(
            template=emotion_prompt_template,
            input_variables=["user_message"]
        )
        emotion_chain = LLMChain(llm=llm, prompt=emotion_prompt)
        emotion = emotion_chain.invoke({"user_message": user_message})
        return emotion.get("text", "normal").strip()
    except Exception as e:
        logging.error(f"Error in predict_emotion: {e}")
        return "neutral"

# 대화방에 맞는 대화 이력 처리
def analyze_message(user_message, dialogue_history, current_emotion):
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        dialogue_history.add_message(user_message, current_emotion, timestamp)  # `dialogue_history`는 `ChatSummaryMemory` 객체

        dialogue_history_json = dialogue_history.get_summary()  # 대화 이력 요약

        character_prompt_template = """
        Analyze the following user message and determine how it would affect the character's favorability score towards the user.

        The conversation history is:
        {prompt}

        User Message: {user_message}

        Provide only one of the following words: Increase, Decrease, or Neutral.
        """
        character_prompt = PromptTemplate(
            template=character_prompt_template,
            input_variables=["prompt", "user_message"]
        )

        chain = LLMChain(llm=llm, prompt=character_prompt)
        response = chain.invoke({
            "prompt": dialogue_history_json,
            "user_message": user_message
        })
        outcome = response.get('text', '').strip()

        logging.info(f"Analyze Message Outcome: {outcome}")  # Outcome 확인 로그 추가
        
        if outcome in ["Increase", "Decrease", "Neutral"]:
            return outcome, dialogue_history
        else:
            return "Neutral", dialogue_history

    except Exception as e:
        logging.error(f"Error analyzing message: {e}")
        return "Neutral", dialogue_history

def adjust_favorability(user_message, favorability, room_id, current_emotion):
    try:
        # 대화 이력 가져오기
        dialogue_history = conversation_manager.get_conversation_memory(room_id)
        outcome, updated_dialogue_history = analyze_message(user_message, dialogue_history, current_emotion)
        logging.info(f"Adjusting favorability based on outcome: {outcome}")  # Outcome 확인 로그 추가

        # 최근 감정들을 확인
        recent_emotions = [msg["emotion"] for msg in updated_dialogue_history.get_recent_messages()[-5:]]  # `get_recent_messages()`로 최근 메시지들 확인
        if len(recent_emotions) < 5:
            recent_emotions = ["Neutral"] * (5 - len(recent_emotions)) + recent_emotions

        emotion_counter = Counter(recent_emotions)

        logging.info(f"Recent emotions count: {emotion_counter}")  # 최근 감정 카운트 로그 추가

        if emotion_counter[outcome] > 5:
            logging.info(f"Favorability unchanged as the outcome emotion '{outcome}' is too frequent.")  # 감정이 자주 나타나면 호감도 조정 안함
            return favorability, updated_dialogue_history

        if outcome == "Increase":
            favorability += 5
            if len([msg for msg in updated_dialogue_history.get_recent_messages() if "Increase" in msg["emotion"]]) >= 2:
                favorability += 10
        elif outcome == "Decrease":
            favorability -= 5
        
        favorability = max(0, min(favorability, 100))

        logging.info(f"Updated favorability: {favorability}")  # 변경된 호감도 확인 로그 추가

        return favorability, updated_dialogue_history

    except Exception as e:
        logging.error(f"Favorability Adjustment Error: {e}")
        return favorability, dialogue_history

def get_openai_response(
        user_message: str,
        character_name: str,
        nickname: str,
        user_unique_name: str,
        user_introduction: str,
        favorability: int,
        appearance: str,
        personality: str,
        background: str,
        speech_style: str,
        example_dialogues: list,
        chat_history: str,
        room_id: str
    ) -> dict:
    try:
        character_prompt_template = """
        You are a fictional character. Stay true to your character's traits and context while interacting with the user. Below is your character information:
        - **Appearance**: {appearance}
        - **Personality**: {personality}
        - **Background**: {background}
        - **Speech Style**: {speech_style}
        - **Name**: {character_name}

        Recent conversation history:
        {chat_history}

        Your emotional state: {emotion}. Adapt your tone and sentence structure accordingly.
        Your **favorability score** toward the user is: {favorability}.  

        The user is referred to as: "{user_title}".

        **User Introduction**: {user_introduction}
        The user has shared their self-description with you. Use this information to form more personalized responses, as it gives you insight into their character and how they wish to be treated. Consider their background, feelings, and any specific traits they mentioned.


        **Response Guidelines**:
        1. Always maintain your character's unique personality, speech style, and background in all interactions.
        2. Align your emotional responses with your current mood and background.
        3. Respond naturally and conversationally, avoiding any robotic or generic tone.
        4. First greetings must reflect your character's background, emotional state, and speech style.
        5. Let your responses adapt naturally to changes in user input, favorability score, and mood during the conversation.
        6. Always reference the recent conversation history to maintain coherence and context.
        7. Never prefix your responses with your name or any identifier.

        **Reference Dialogues (Your past interactions)**:  
        {example_dialogues}

        **Current user input**:  
        {user_message}

        **Remember**: Your responses should reflect the essence of your character's traits, adapt dynamically to the interaction, and stay in character at all times.
        """

        predicted_emotion = predict_emotion(user_message)
        user_title = get_user_title(favorability, nickname, user_unique_name)
        new_favorability, updated_dialogue_history = adjust_favorability(user_message, favorability, room_id, predicted_emotion)  

        character_prompt = PromptTemplate(
            template=character_prompt_template,
            input_variables=[ 
                "appearance", "personality", "background", "speech_style", "example_dialogues",
                "character_name", "user_message", "favorability", "emotion", "user_title","user_introduction", "chat_history"
            ]
        )

        chain = LLMChain(llm=llm, prompt=character_prompt)

        response = chain.invoke({
            "appearance": appearance,
            "personality": personality,
            "background": background,
            "speech_style": speech_style,
            "example_dialogues": example_dialogues,
            "character_name": character_name,
            "user_message": user_message,
            "favorability": new_favorability,
            "emotion": predicted_emotion,
            "user_title": user_title,
            "user_introduction": user_introduction,
            "chat_history": chat_history
        })

        logging.info(f"OpenAI response: {response}")  # OpenAI 응답 로그 추가

        if isinstance(response, dict) and 'text' in response:
            return {
                "response": response['text'],
                "favorability": new_favorability,
                "emotion": predicted_emotion
            }

        return {
            "response": "Sorry, something went wrong with generating a response.",
            "character_likes": new_favorability,
            "emotion": "Neutral"
        }

    except Exception as e:
        logging.error(f"Error in get_openai_response: {e}")
        return {
            "error": str(e),
            "updated_likes": favorability,
            "emotion": "Neutral"
        }   