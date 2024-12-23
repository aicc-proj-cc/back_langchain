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


# 사용자 호칭 설정
def get_user_title(character_likes):
    """
    호감도에 따라 사용자에 대한 호칭을 설정합니다.
    """
    if character_likes < 30:
        return "손님"
    elif character_likes < 70:
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
    # print("emotion :", emotion)
    # print("current emotion :", emotion.get("text", "Neutral"))
    return emotion.get("text", "Neutral")

# 호감도 조정 함수
def adjust_likability(user_message, character_likes):
    """
    사용자의 메시지에 따라 캐릭터의 호감도를 조정합니다.
    """
    if "고마워" in user_message or "좋아" in user_message:
        character_likes += 10
    elif "싫어" in user_message or "실망" in user_message or "헤어져" in user_message:
        character_likes -= 10
    return max(0, min(character_likes, 100))  # 호감도를 0~100 사이로 제한


# OpenAI 응답 생성 함수
def get_openai_response(prompt: str, user_message: str, character_name: str, character_likes: int) -> str:
    """
    OpenAI의 API를 사용하여 사용자 입력에 따라 캐릭터 챗봇이 응답을 생성

    Parameters:
        prompt (str): 캐릭터와 관련된 프롬프트 정보.
        user_message (str): 사용자가 입력한 메시지.
        character_name (str): 캐릭터 이름.
        character_likes (int): 캐릭터 호감도.

    Returns:
        str: GPT 모델이 생성한 응답.
    """
    # print('prompt :', prompt)
    # print('character :', character_name)
    # print('user_input :', user_message)

    # 캐릭터에 해당하는 프롬프트 템플릿 생성
    character_prompt_template = """
    You are a fictional character. The information about the character is as follows:
    {prompt}.
    Act as if you are the actual character, and respond to the user's message in a manner that reflects your personality and current mood.

    The character's name is {character_name}.
    The character's mood is {emotion}.
    The character refers to the user as "{user_title}".
    The character's current likability score toward the user is {character_likes}.

    User input: {user_message}
    """

    # 템플릿을 채워서 완성된 프롬프트 생성
    character_prompt = PromptTemplate(
        template=character_prompt_template,
        input_variables=["prompt", "character_name", "user_message", "character_likes", "emotion", "user_title"]
    )

    # LLMChain 생성
    chain = LLMChain(llm=llm, prompt=character_prompt)

    new_character_likes = adjust_likability(user_message, character_likes)

    try:
        # 체인을 통해 결과 생성
        response = chain.invoke({
            "prompt": prompt,
            "character_name": character_name,
            "user_message": user_message,
            "character_likes": new_character_likes,
            "emotion": predict_emotion(user_message),
            "user_title": get_user_title(character_likes)
        })
        return response # OpenAI API 응답 내용 반환
    except Exception as e:
        return f"Error: {str(e)}"


