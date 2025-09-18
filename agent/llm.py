from typing import List, Dict, Optional
import os
from config.settings import settings

try:
    from openai import OpenAI, AsyncOpenAI
except Exception:  # pragma: no cover
    OpenAI = None
    AsyncOpenAI = None

class LLM:
    """Lightweight wrapper around OpenAI Chat Completions (v1+)."""
    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None):
        self.model = model or settings.MODEL_NAME or "gpt-4o-mini"
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self._sync = OpenAI(api_key=self.api_key) if (self.api_key and OpenAI) else None
        self._async = AsyncOpenAI(api_key=self.api_key) if (self.api_key and AsyncOpenAI) else None

    async def complete(self, messages: List[Dict], temperature: float = 0.2) -> str:
        if not self.api_key or not self._async:
            raise RuntimeError("OPENAI_API_KEY 未配置或 openai 未安装：无法调用聊天模型。pip install openai，并在 .env 设置 OPENAI_API_KEY。")
        resp = await self._async.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
        )
        return (resp.choices[0].message.content or "").strip()

    def complete_sync(self, messages: List[Dict], temperature: float = 0.2) -> str:
        if not self.api_key or not self._sync:
            raise RuntimeError("OPENAI_API_KEY 未配置或 openai 未安装：无法调用聊天模型。pip install openai，并在 .env 设置 OPENAI_API_KEY。")
        resp = self._sync.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
        )
        return (resp.choices[0].message.content or "").strip()
