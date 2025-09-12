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
    mcp_clients = []
    router = None

    def _split_commands(cmd):
        # 支持：字符串用 ; 分隔、多行；或传 list/list[list]
        if isinstance(cmd, str):
            parts = [p.strip() for p in cmd.replace("\n", ";").split(";") if p.strip()]
            return [[*p.split(" ", 1)][0], *([p.split(" ", 1)[1]] if " " in p else [])] and [
                p.split(" ") for p in parts
            ]
        if isinstance(cmd, (list, tuple)):
            # 如果是 list[list] 就原样返回；如果是 list[str] 就包一层
            if cmd and isinstance(cmd[0], (list, tuple)):
                return list(cmd)
            return [list(cmd)]
        return []

    try:
        cmds = _split_commands(cfg.mcp.command)
        for argv in cmds:
            c = MCPClient(argv)  # 这里的 MCPClient 应该支持 list[str] 作为命令
            await c.start()
            await c.initialize({"client": "llm-mcp-rag-min", "version": "0.1"})
            mcp_clients.append(c)

        # 组合路由器：依次尝试每个 MCP，哪个有这个 tool 就用哪个
        class CombinedRouter:
            def __init__(self, clients):
                self.clients = clients

            async def refresh(self):  # 和原接口保持一致
                return

            async def call(self, name, arguments):
                last_err = None
                for c in self.clients:
                    try:
                        return await c.tools_call(name, arguments)
                    except Exception as e:
                        msg = str(e).lower()
                        if "未知工具" in msg or "unknown tool" in msg:
                            last_err = e
                            continue
                        last_err = e
                        break
                if last_err:
                    raise last_err
                raise RuntimeError("No MCP client available.")

            @property
            def mcp(self):
                return None  # 兼容老代码不直接用到

        if len(mcp_clients) == 1:
            router = ToolRouter(mcp_clients[0])
            await router.refresh()
        elif len(mcp_clients) > 1:
            router = CombinedRouter(mcp_clients)
        else:
            router = None
    except Exception as e:
        print("⚠️ MCP 初始化失败，将不使用工具。原因：", repr(e))
        for c in mcp_clients:
            try:
                await c.stop()
            except Exception:
                pass
        mcp_clients = []
        router = None

    # Orchestrator
    orch = Orchestrator(llm, retr, mem, router, rag_top_k=cfg.rag.top_k)

    print("🤖 学习助教 Agent 已就绪！")
    print("直接输入问题 → 知识问答")
    print("/explain <主题> → 讲解模式")
    print("/solve <题目> → 解题模式")
    print("/run <lang> <code or ```...```> [--timeout=秒] → 运行代码（python/c/cpp/java）")  # ← 新增
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
        elif q.startswith("/run "):
            mode = "run"
            q = q[len("/run "):].strip()
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
