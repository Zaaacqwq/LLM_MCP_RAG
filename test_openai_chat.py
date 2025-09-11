# python -u test_openai_chat.py
import os, httpx, json, asyncio

API_KEY = ""  # 仅临时测试，用完删
MODEL = "gpt-4o-mini"              # 建议先用这个小模型测试

async def main():
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": MODEL,
        "messages": [{"role":"user","content":"只回复数字: 8+5 等于几？"}],
        "temperature": 0,
    }
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post("https://api.openai.com/v1/chat/completions",
                              headers=headers, data=json.dumps(payload))
        print("status:", r.status_code)
        print(r.text)

if __name__ == "__main__":
    asyncio.run(main())
