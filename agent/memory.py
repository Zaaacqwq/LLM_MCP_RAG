from dataclasses import dataclass, field

@dataclass
class Memory:
    history: list[dict] = field(default_factory=list)   # chat history
    tool_cache: dict = field(default_factory=dict)      # 你可以缓存工具结果（可选）

    def add_user(self, content: str):
        self.history.append({"role": "user", "content": content})

    def add_assistant(self, content: str):
        self.history.append({"role": "assistant", "content": content})

    def context(self, max_turns=6):
        return self.history[-max_turns*2:]  # 简单截断
