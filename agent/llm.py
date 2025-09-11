import os, httpx, json
from .config import LLMConfig

class LLM:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self.api_key = cfg.api_key or os.getenv("OPENAI_API_KEY", "")

    async def complete(self, messages: list[dict], temperature: float = 0.2):
        if self.cfg.provider != "openai":
            raise NotImplementedError("Only OpenAI demo here.")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": temperature,
        }
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(f"{self.cfg.api_base}/chat/completions",
                                  headers=headers, data=json.dumps(payload))
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
