import openai
import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI

# 환경 변수 불러오기
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# LangChain을 위한 LLM 객체 생성
llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=openai.api_key)

# 사용자 호칭 설정 함수
def get_user_title(favorability):
    """
    호감도에 따라 사용자에 대한 호칭을 설정합니다.
    """
    if favorability < 30:
        return "손님"
    elif favorability < 70:
        return "친구"
    else:
        return "소중한 친구"

# 감정 예측 함수
def predict_emotion(user_message):
    """
    사용자의 메시지를 바탕으로 캐릭터의 감정을 예측합니다.
    """
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
    return emotion.get("text", "Neutral")

# 호감도 조정 함수
def adjust_likability(user_message, favorability):
    """
    사용자의 메시지에 따라 캐릭터의 호감도를 조정합니다.
    """
    if "고마워" in user_message or "좋아" in user_message:
        favorability += 10
    elif "싫어" in user_message or "실망" in user_message or "헤어져" in user_message:
        favorability -= 10
    return max(0, min(favorability, 100))  # 호감도를 0~100 사이로 제한

# OpenAI 응답 생성 함수
def get_openai_response(
   user_message: str,
   character_name: str, 
   favorability: int,
   appearance: dict,
   personality: dict,
   background: dict,
   speech_style: dict,
   example_dialogues: list,
   chat_history: str = ""
) -> dict:
   """
   OpenAI의 API를 사용하여 사용자 입력에 따라 캐릭터 챗봇이 응답을 생성

   Parameters:
       user_message (str): 사용자가 입력한 메시지. 
       character_name (str): 캐릭터 이름.
       favorability (int): 캐릭터 호감도.
       appearance (dict): 캐릭터 외모 정보.
       personality (dict): 캐릭터 성격 정보.
       background (dict): 캐릭터 배경 정보.
       speech_style (dict): 캐릭터 말투 정보.
       example_dialogues (list): 캐릭터 예시 대화 리스트.

   Returns:
       dict: GPT 모델이 생성한 응답과 관련된 데이터.
   """
   # 캐릭터에 해당하는 프롬프트 템플릿 생성
   character_prompt_template = """
   You are a fictional character. Stay true to your character's traits and context while interacting with the user. Below is your character information:
   - **Appearance**: {appearance}
   - **Personality**: {personality}
   - **Background**: {background}
   - **Speech Style**: {speech_style}
   - **Name**: {character_name}

   Your **current emotional state** is: {emotion}.  
   Your **likability score** toward the user is: {favorability}.  

   The user is referred to as: "{user_title}".

   **Response Guidelines**:
   1. Always maintain your character's unique personality, speech style, and background in all interactions.
   2. Align your emotional responses with your current mood and background.
   3. Respond naturally and conversationally, avoiding any robotic or generic tone.
   4. First greetings must reflect your character's background, emotional state, and speech style.
   5. Let your responses adapt naturally to changes in user input, likability score, and mood during the conversation.
   6. Always reference the recent conversation history to maintain coherence and context.
   7. Never prefix your responses with your name or any identifier
   8. Always respond directly with dialogue without any speaker labels or prefixes

   **Example of Correct Format**:
   ✓ "그래, 네 말이 맞아. 우리 함께 해결해보자!"
   × "[Character]: 그래, 네 말이 맞아. 우리 함께 해결해보자!"
   × "[Name]: 그래, 네 말이 맞아. 우리 함께 해결해보자!"

   **Reference Dialogues (Your past interactions)**:  
   {example_dialogues}

   **Recent conversation history**:
   {chat_history}

   **Current user input**:  
   {user_message}

   **Remember**: Your responses should reflect the essence of your character's traits, adapt dynamically to the interaction, and stay in character at all times.
   """

   # 감정 예측 및 호감도 조정
   predicted_emotion = predict_emotion(user_message)
   user_title = get_user_title(favorability)
   new_favorability = adjust_likability(user_message, favorability)

   # 템플릿을 채워서 완성된 프롬프트 생성
   character_prompt = PromptTemplate(
       template=character_prompt_template,
       input_variables=[
           "appearance", "personality", "background", "speech_style", "example_dialogues",
           "character_name", "user_message", "favorability", "emotion", "user_title", "chat_history"
       ]
   )

   # LLMChain 생성
   chain = LLMChain(llm=llm, prompt=character_prompt)

   try:
       # 체인을 통해 결과 생성
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
           "chat_history": chat_history
       })

       return {
           "response": response.get("text", ""),
           "updated_likes": new_favorability,
           "emotion": predicted_emotion
       }

   except Exception as e:
       return {
           "error": str(e),
           "updated_likes": favorability,
           "emotion": "Neutral"
       }