from .memory import Memory
from .retriever import Retriever
from .llm import LLM

SYS_PROMPT = """You are a minimal agent. 
- Use RAG context when provided.
- If a tool result is present, incorporate it.
- Be concise and factual in Chinese."""

def build_messages(mem: Memory, rag_context: list[str] | None, tool_obs: str | None):
    msgs = [{"role":"system","content":SYS_PROMPT}]
    ctx = ""
    if rag_context:
        ctx += "\n[RAG Context]\n" + "\n---\n".join(rag_context[:4])
    if tool_obs:
        ctx += f"\n[Tool Result]\n{tool_obs}"
    if ctx:
        msgs.append({"role":"system","content":ctx})
    msgs.extend(mem.context())
    return msgs

class Orchestrator:
    def __init__(self, llm: LLM, retriever: Retriever, memory: Memory, router, rag_top_k=4):
        self.llm = llm
        self.retriever = retriever
        self.memory = memory
        self.router = router
        self.rag_top_k = rag_top_k

    async def step(self, user_text: str):
        self.memory.add_user(user_text)

        # 1) RAG 检索
        rag_ctx = []
        if self.rag_top_k and self.rag_top_k > 0:
            rag_ctx = await self.retriever.topk(user_text, self.rag_top_k)

        # 2) 工具路由（MCP）
        tool_obs = None
        if self.router and self.router.need_tool(user_text):
            name, args = self.router.pick(user_text)
            if name:
                res = await self.router.mcp.tools_call(name, args or {})
                tool_obs = str(res)

        # 3) LLM 生成
        messages = build_messages(self.memory, rag_ctx, tool_obs)
        reply = await self.llm.complete(messages)
        self.memory.add_assistant(reply)
        return reply
