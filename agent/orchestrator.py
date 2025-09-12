# agent/orchestrator.py
import re
import json
import ast

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

    # 拼接工具结果（给 LLM 参考或解释）
    if tool_obs:
        msgs.append({"role": "system", "content": f"[Tool Result]\n{tool_obs}"})

    # 加上历史对话
    msgs.extend(mem.context())
    return msgs


async def _tools_call(router, name: str, args: dict):
    """
    统一的工具调用入口：
    - 优先 ToolRouter.call / invoke
    - 否则退回到底层 mcp.tools_call
    """
    if router is None:
        raise RuntimeError("工具不可用：未连接 MCP。")
    if hasattr(router, "call"):
        return await router.call(name, args)
    if hasattr(router, "invoke"):
        return await router.invoke(name, args)
    if getattr(router, "mcp", None) and hasattr(router.mcp, "tools_call"):
        return await router.mcp.tools_call(name, args)
    raise RuntimeError("无法调用工具：未发现可用的调用方法。")


def _parse_matrix_side(text: str):
    """把一侧矩阵字符串解析为 Python 列表（优先 JSON，其次 literal_eval）"""
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        return ast.literal_eval(text)


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

        # 2) 工具调用（仅在 solve 模式才尝试直连工具；其它模式交给 LLM）
        tool_obs = None
        if mode == "solve" and self.router:
            txt = user_text.strip()

            try:
                # ---------- 画图：画图 <expr> 从 <start> 到 <end> ----------
                m = re.match(r"^画图\s+(.+?)\s+从\s+(-?\d+(?:\.\d+)?)\s+到\s+(-?\d+(?:\.\d+)?)\s*$", txt)
                if m:
                    expr, start, end = m.group(1), float(m.group(2)), float(m.group(3))
                    res = await _tools_call(self.router, "math.plot",
                                            {"expr": expr, "var": "x", "start": start, "end": end, "num": 600})
                    path = res.get("path") or res.get("image_path")
                    tool_obs = f"[math.plot] {json.dumps(res, ensure_ascii=False)}"
                    # 直接把路径返回到对话里更友好
                    self.memory.add_assistant(f"✅ 已生成图像：{path}")
                    # 不阻断后续回答：继续把 tool_obs 给到 LLM 用于补充说明

                # ---------- 矩阵乘法：矩阵乘法/计算 <A> [*|×|x] <B> ----------
                if tool_obs is None:
                    m2 = re.match(
                        r"^(?:矩阵乘法|计算)\s+(\[.*\])\s*(?:\*|×|x)\s*(\[.*\])\s*$",
                        txt.replace(" ", ""))
                    # 去空格后再解析，以便容忍空格
                    if not m2:
                        # 再来一版宽松匹配（保留空格）
                        m2 = re.match(r"^(?:矩阵乘法|计算)\s+(\[.*\])\s*(?:\*|×|x)\s*(\[.*\])\s*$", txt)
                    if m2:
                        A = _parse_matrix_side(m2.group(1))
                        B = _parse_matrix_side(m2.group(2))
                        res = await _tools_call(self.router, "math.matrix_multiply", {"A": A, "B": B})
                        tool_obs = f"[math.matrix_multiply] {json.dumps(res, ensure_ascii=False)}"
                        mat = res.get("matrix") or res.get("result")
                        shape = res.get("shape")
                        self.memory.add_assistant(f"✅ 矩阵乘法结果（shape={shape}）：\n{mat}")

                # ---------- 求导：对 <expr> 求导 ----------
                if tool_obs is None:
                    m3 = re.match(r"^对\s+(.+?)\s+求导\s*$", txt)
                    if m3:
                        expr = m3.group(1)
                        res = await _tools_call(self.router, "math.diff", {"expr": expr, "var": "x"})
                        tool_obs = f"[math.diff] {json.dumps(res, ensure_ascii=False)}"
                        self.memory.add_assistant(f"∂/∂x {expr} = {res.get('derivative')}")

                # ---------- 积分：对 <expr> 积分 ----------
                if tool_obs is None:
                    m4 = re.match(r"^对\s+(.+?)\s+积分\s*$", txt)
                    if m4:
                        expr = m4.group(1)
                        res = await _tools_call(self.router, "math.integrate", {"expr": expr, "var": "x"})
                        tool_obs = f"[math.integrate] {json.dumps(res, ensure_ascii=False)}"
                        self.memory.add_assistant(f"∫ {expr} dx = {res.get('integral')} + C")

                # ---------- 解方程（兜底） ----------
                if tool_obs is None:
                    # 更严格的“像方程”的判据：包含 '=' 或 明确“解方程/求解”
                    looks_like_equation = ("=" in txt) or ("解方程" in txt) or ("求解" in txt)
                    if looks_like_equation:
                        res = await _tools_call(self.router, "math.solve_equation", {"expr": txt, "var": "x"})
                        tool_obs = f"[math.solve_equation] {json.dumps(res, ensure_ascii=False)}"
                        sols = res.get("solutions", [])
                        self.memory.add_assistant(f"解：{', '.join(map(str, sols)) if sols else '(无解或非代数方程)'}")

            except Exception as e:
                tool_obs = f"[tool_error] {e}"

        # 3) LLM 生成
        messages = build_messages(self.memory, rag_ctx, tool_obs, mode)
        reply = await self.llm.complete(messages)
        self.memory.add_assistant(reply)
        return reply
