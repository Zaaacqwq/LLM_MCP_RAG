from typing import List
from .schema import Message

class Memory:
    def __init__(self):
        self.messages: List[Message] = []

    def add_user(self, text: str):
        self.messages.append(Message(role="user", content=text))

    def add_assistant(self, text: str, meta=None):
        self.messages.append(Message(role="assistant", content=text, meta=meta or {}))

    def last(self, n: int = 5) -> List[Message]:
        return self.messages[-n:]
