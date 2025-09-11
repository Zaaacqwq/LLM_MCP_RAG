import os, httpx, json, numpy as np

class Embeddings:
    def __init__(self, model="text-embedding-3-small", api_key: str | None=None):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")

    async def embed(self, texts: list[str]) -> list[list[float]]:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {"model": self.model, "input": texts}
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post("https://api.openai.com/v1/embeddings",
                                  headers=headers, data=json.dumps(payload))
            r.raise_for_status()
            data = r.json()["data"]
            return [d["embedding"] for d in data]

def cosine_sim(a, b):
    a = np.array(a); b = np.array(b)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-12
    return float(np.dot(a, b) / denom)
