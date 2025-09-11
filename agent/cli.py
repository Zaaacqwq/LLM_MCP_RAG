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
    cfg = AppConfig()

    # 初始化 LLM & Embeddings
    llm = LLM(cfg.llm)
    emb_api_key = cfg.llm.api_key or os.getenv("OPENAI_API_KEY", "")
    emb = Embeddings(model="text-embedding-3-small", api_key=emb_api_key)

    retr = Retriever(cfg.rag, emb)
    mem = Memory()

    # MCP（可选）
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

    # Orchestrator
    orch = Orchestrator(llm, retr, mem, router, rag_top_k=cfg.rag.top_k)

    print("🤖 学习助教 Agent 已就绪！")
    print("直接输入问题 → 知识问答")
    print("/explain <主题> → 讲解模式")
    print("/solve <题目> → 解题模式")
    print("/exit → 退出")

    while True:
        try:
            q = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Bye")
            break

        if q in ("/exit", "exit", "quit"):
            break

        # 解析模式
        if q.startswith("/explain "):
            mode = "explain"
            q = q[len("/explain "):].strip()
        elif q.startswith("/solve "):
            mode = "solve"
            q = q[len("/solve "):].strip()
        else:
            mode = "qa"

        try:
            ans = await orch.step(q, mode=mode)
            print(f"\nAgent> {ans}")
        except Exception as e:
            print("Error:", repr(e))
            traceback.print_exc()

    if mcp:
        try:
            await mcp.stop()
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(main())
