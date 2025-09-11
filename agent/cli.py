# agent/cli.py
import os
import asyncio
import traceback

from agent.config import AppConfig
from agent.llm import LLM
from agent.embeddings import Embeddings
from agent.retriever import Retriever
from agent.memory import Memory
from agent.mcp_client import MCPClient
from agent.tool_router import ToolRouter
from agent.orchestrator import Orchestrator


async def main():
    # 加载配置
    cfg = AppConfig()

    # --- LLM & Embeddings ---
    llm = LLM(cfg.llm)

    # 关键：把 config 里的 api_key 也传给 Embeddings（否则只会读环境变量）
    emb_api_key = cfg.llm.api_key or os.getenv("OPENAI_API_KEY", "")
    emb = Embeddings(model="text-embedding-3-small", api_key=emb_api_key)

    retr = Retriever(cfg.rag, emb)
    mem = Memory()

    # --- 可选：MCP 工具端 ---
    mcp = None
    router = None
    if cfg.mcp.command:
        try:
            mcp = MCPClient(cfg.mcp.command)
            await mcp.start()
            await mcp.initialize({"client": "llm-mcp-rag-min", "version": "0.1"})
            router = ToolRouter(mcp)
            await router.refresh()
        except Exception as e:
            print("⚠️ MCP 初始化失败，将不使用工具。原因：", repr(e))
            mcp = None
            router = None

    # --- Orchestrator ---
    # 先用 rag_top_k=0 验证直连；稳定后改回 cfg.rag.top_k（例如 2 或 4）
    rag_top_k = cfg.rag.top_k  # 如需临时关闭 RAG，可改成 0
    orch = Orchestrator(llm, retr, mem, router, rag_top_k=rag_top_k)

    print("🤖 Minimal Agent ready. 输入你的问题，/exit 退出。")

    # 交互循环
    while True:
        try:
            q = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Bye")
            break

        if q in ("/exit", "exit", "quit"):
            break

        try:
            ans = await orch.step(q)
            print(f"\nAgent> {ans}")
        except Exception as e:
            # 打印完整堆栈，便于快速定位
            print("Error:", repr(e))
            traceback.print_exc()

    # 退出前清理 MCP
    if mcp:
        try:
            await mcp.stop()
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(main())
