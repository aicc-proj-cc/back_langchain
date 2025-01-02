import json
from collections import deque

class ChatSummaryMemory:
    def __init__(self, max_history_length=5):
        self.max_history_length = max_history_length
        self.history = deque(maxlen=max_history_length)

    def add_message(self, message, emotion, timestamp):
        self.history.append({
            "message": message,
            "emotion": emotion,
            "timestamp": timestamp
        })

    def get_summary(self):
        # 대화 이력을 요약하는 로직을 추가
        # 예를 들어 최근 5개의 메시지만 요약하거나 특정 기준으로 필터링할 수 있음
        return json.dumps(list(self.history), ensure_ascii=False, indent=4)
    
    def get_recent_messages(self):
        # 최근 메시지들을 가져오는 메소드
        return list(self.history)
