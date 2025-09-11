# agent/orchestrator.py
from agent.memory import Memory
from agent.retriever import Retriever
from agent.llm import LLM

QA_PROMPT = """你是学习助教，用中文回答问题。
- 只基于提供的[RAG]片段。
- 若片段不足，请明确说“不足”。"""

EXPLAIN_PROMPT = """你是学习助教，用中文讲解知识点。
- 基于[RAG]片段。
- 给出：直白解释、关键公式/定义、类比举例、常见误区。"""

SOLVE_PROMPT = """你是学习助教，用中文解答习题。
- 先写思路，再分步骤推导，最后给出答案。
- 若[RAG]片段不足，可用常识或数学知识补充，但要标明。"""


def build_messages(mem: Memory, rag_ctx: list[dict] | None, tool_obs: str | None, mode: str):
    # 选择不同的系统提示词
    if mode == "qa":
        sys_prompt = QA_PROMPT
    elif mode == "explain":
        sys_prompt = EXPLAIN_PROMPT
    elif mode == "solve":
        sys_prompt = SOLVE_PROMPT
    else:
        sys_prompt = QA_PROMPT

    msgs = [{"role": "system", "content": sys_prompt}]

    # 拼接 RAG 片段
    if rag_ctx:
        ctx = "\n".join(
            [
                f"({i+1}) 来自: {meta.get('source')}#{meta.get('chunk_id')}\n{meta['text']}"
                for i, meta in enumerate(rag_ctx)
            ]
        )
        msgs.append({"role": "system", "content": "[RAG]\n" + ctx})

    # 拼接工具结果（暂时没用，但保留接口）
    if tool_obs:
        msgs.append({"role": "system", "content": f"[Tool Result]\n{tool_obs}"})

    # 加上历史对话
    msgs.extend(mem.context())
    return msgs


class Orchestrator:
    def __init__(self, llm: LLM, retriever: Retriever, memory: Memory, router, rag_top_k=4):
        self.llm = llm
        self.retriever = retriever
        self.memory = memory
        self.router = router
        self.rag_top_k = rag_top_k

    async def step(self, user_text: str, mode="qa"):
        self.memory.add_user(user_text)

        # 1) RAG 检索
        rag_ctx = []
        if self.rag_top_k and self.rag_top_k > 0:
            rag_ctx = await self.retriever.topk(user_text, self.rag_top_k)

        # 2) 工具调用（仅在 solve 模式）
        tool_obs = None
        if mode == "solve" and self.router and self.router.mcp:
            txt = user_text.replace(" ", "")
            try:
                if "=" in txt or "^" in txt or "x" in txt:
                    res = await self.router.mcp.tools_call(
                        "math.solve_equation", {"expr": user_text, "var": "x"}
                    )
                    tool_obs = f"[math.solve_equation] {res}"
                elif "求导" in user_text:
                    res = await self.router.mcp.tools_call(
                        "math.diff", {"expr": user_text, "var": "x"}
                    )
                    tool_obs = f"[math.diff] {res}"
                elif "积分" in user_text:
                    res = await self.router.mcp.tools_call(
                        "math.integrate", {"expr": user_text, "var": "x"}
                    )
                    tool_obs = f"[math.integrate] {res}"
            except Exception as e:
                tool_obs = f"[tool_error] {e}"

        # 3) LLM 生成
        messages = build_messages(self.memory, rag_ctx, tool_obs, mode)
        reply = await self.llm.complete(messages)
        self.memory.add_assistant(reply)
        return reply
