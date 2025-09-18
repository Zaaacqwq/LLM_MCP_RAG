import asyncio, sys
from rich.console import Console
from .memory import Memory
from .orchestrator import Orchestrator

HELP = """
直接输入问题 → 知识问答
/explain <主题> → 讲解模式
/solve <题目> → 解题模式
/run <lang> <code or ```...```> [--timeout=秒] → 运行代码（python）
/reindex → 读取文档并用本地 chunker 重建索引
/exit → 退出
"""

async def main():
    cons = Console()
    cons.print("🤖 学习助教 Agent 已就绪！\n" + HELP)
    mem = Memory()
    orch = Orchestrator(mem)
    while True:
        try:
            q = input("\nYou> ").strip()
        except EOFError:
            break
        if not q:
            continue
        if q == "/exit":
            break
        mode = None
        if q.startswith(("/explain","/solve","/run","/reindex")):
            parts = q.split(maxsplit=1)
            mode = parts[0][1:]
            payload = parts[1] if len(parts)>1 else ""
            text = payload if mode != "run" else q  # run 需要原始行
        else:
            text = q
        res = await orch.step(q if mode=="run" else text, mode=mode)
        cons.print(res.final_text)
        if res.errors:
            cons.print(f"[red]Errors:[/red] {res.errors}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
