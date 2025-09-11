# build_index.py
import asyncio
import os
from agent.config import AppConfig
from agent.embeddings import Embeddings
from agent.retriever import Retriever

async def main():
    cfg = AppConfig()
    emb_api_key = cfg.llm.api_key or os.getenv("OPENAI_API_KEY", "")
    emb = Embeddings(model="text-embedding-3-small", api_key=emb_api_key)
    retr = Retriever(cfg.rag, emb)

    # 强制重建索引
    if cfg.rag.index_path.exists():
        print("删除旧索引:", cfg.rag.index_path)
        cfg.rag.index_path.unlink()

    print("开始构建索引...")
    await retr.ensure_index()
    print("✅ 索引已完成:", cfg.rag.index_path)

if __name__ == "__main__":
    asyncio.run(main())
