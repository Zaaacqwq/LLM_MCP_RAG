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
    if mode == "qa":
        sys_prompt = QA_PROMPT
    elif mode == "explain":
        sys_prompt = EXPLAIN_PROMPT
    elif mode == "solve":
        sys_prompt = SOLVE_PROMPT
    else:
        sys_prompt = QA_PROMPT

    msgs = [{"role": "system", "content": sys_prompt}]

    if rag_ctx:
        ctx = "\n".join(
            [
                f"({i+1}) 来自: {meta.get('source')}#{meta.get('chunk_id')}\n{meta['text']}"
                for i, meta in enumerate(rag_ctx)
            ]
        )
        msgs.append({"role": "system", "content": "[RAG]\n" + ctx})

    if tool_obs:
        msgs.append({"role": "system", "content": f"[Tool Result]\n{tool_obs}"})

    msgs.extend(mem.context())
    return msgs


async def _tools_call(router, name: str, args: dict):
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
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        return ast.literal_eval(text)


def _strip_code_block(code: str) -> str:
    code = code.strip()
    if code.startswith("```") and code.endswith("```") and len(code) >= 6:
        return code[3:-3]
    return code


async def _run_mode_dispatch(router, payload: str):
    """
    解析并执行 run 模式：
      语法：<lang> <code or ```...```> [--timeout=秒] [--verbose]
      语言：python/py, c, cpp/c++/cc, java
    默认输出尽量精简；加 --verbose 输出完整细节。
    """
    import re

    # 提取可选 verbose / timeout
    verbose = False
    if "--verbose" in payload or "-v" in payload.split():
        verbose = True
        payload = payload.replace("--verbose", "").replace("-v", "")

    timeout = None
    m_to = re.search(r"--timeout\s*=\s*([0-9]+(?:\.[0-9]+)?)", payload)
    if m_to:
        try:
            timeout = float(m_to.group(1))
        except Exception:
            pass
        payload = payload[:m_to.start()] + payload[m_to.end():]

    payload = payload.strip()
    if not payload:
        return "用法：/run <python|c|cpp|java> <code or ```...```> [--timeout=秒] [--verbose]"

    parts = payload.split(None, 1)
    lang = parts[0].lower()
    code = parts[1].strip() if len(parts) > 1 else ""

    if lang in ("python", "py"):
        tool = "python.run_code"
    elif lang == "c":
        tool = "c.run_code"
    elif lang in ("cpp", "c++", "cc"):
        tool = "cpp.run_code"
    elif lang == "java":
        tool = "java.run_code"
    else:
        return "不支持的语言，请使用：python/py、c、cpp、java。"

    if not code:
        return "请在同一行提供代码，或在同一条输入中用三引号包裹：```...```"

    # 去除同一条输入内闭合的三引号
    def _strip_code_block(s: str) -> str:
        s = s.strip()
        if s.startswith("```") and s.endswith("```") and len(s) >= 6:
            return s[3:-3]
        return s

    code = _strip_code_block(code)
    code = code.replace("\\n", "\n").replace("\\t", "\t")

    args = {"code": code}
    if timeout is not None:
        args["timeout"] = timeout

    res = await _tools_call(router, tool, args)

    # ---------- 精简/详细 两种格式 ----------
    def _indent(text: str, pad: str = "    ") -> str:
        if text is None:
            text = ""
        if not text.endswith("\n"):
            text += "\n"
        return "".join(pad + ln for ln in text.splitlines(True))

    def fmt_section(title, obj):
        if not obj:
            return ""
        return (
            f"{title}:\n"
            f"  exit_code: {obj.get('exit_code')}\n"
            f"  time_ms:   {obj.get('time_ms')}\n"
            f"  stdout:\n{_indent(obj.get('stdout',''))}\n"
            f"  stderr:\n{_indent(obj.get('stderr',''))}\n"
        )

    if verbose:
        # 详细模式：保留完整结构信息
        if tool == "python.run_code":
            out = [
                f"🟩 {tool}",
                f"workdir: {res.get('workdir','')}",
                f"files:   {', '.join(res.get('files', []))}",
                fmt_section("run", res),
            ]
        else:
            out = [
                f"🟩 {tool}",
                f"workdir: {res.get('workdir','')}",
                f"files:   {', '.join(res.get('files', []))}",
                fmt_section("compile", res.get("compile")),
            ]
            if res.get("compile", {}).get("exit_code") == 0:
                out.append(fmt_section("run", res.get("run")))
        return "\n".join(s for s in out if s)

    # 精简模式：只给最关心的结果
    if tool == "python.run_code":
        rc = res.get("exit_code", 0)
        if rc == 0:
            return res.get("stdout", "").rstrip("\n") or "(无输出)"
        else:
            # 失败：给最关键的错误（stderr 的第一段）
            err = (res.get("stderr") or "").strip()
            return f"运行失败 (exit={rc}): {err.splitlines()[0] if err else 'unknown error'}"
    else:
        comp = res.get("compile", {}) or {}
        if comp.get("exit_code", 0) != 0:
            err = (comp.get("stderr") or "").strip()
            return f"编译失败 (exit={comp.get('exit_code')}): {err.splitlines()[0] if err else 'unknown error'}"
        run = res.get("run", {}) or {}
        rc = run.get("exit_code", 0)
        if rc == 0:
            return run.get("stdout", "").rstrip("\n") or "(无输出)"
        else:
            err = (run.get("stderr") or "").strip()
            return f"运行失败 (exit={rc}): {err.splitlines()[0] if err else 'unknown error'}"

class Orchestrator:
    def __init__(self, llm: LLM, retriever: Retriever, memory: Memory, router, rag_top_k=4):
        self.llm = llm
        self.retriever = retriever
        self.memory = memory
        self.router = router
        self.rag_top_k = rag_top_k

    async def step(self, user_text: str, mode="qa"):
        self.memory.add_user(user_text)

        # --- 新增：run 模式（不走 LLM） ---
        if mode == "run":
            if not self.router:
                msg = "工具不可用：未连接 MCP。请在配置中启用 code_server。"
                self.memory.add_assistant(msg)
                return msg
            try:
                msg = await _run_mode_dispatch(self.router, user_text)
            except Exception as e:
                msg = f"运行失败：{e}"
            self.memory.add_assistant(msg)
            return msg

        # 1) RAG 检索
        rag_ctx = []
        if self.rag_top_k and self.rag_top_k > 0:
            rag_ctx = await self.retriever.topk(user_text, self.rag_top_k)

        # 2) 工具调用（仅在 solve 模式：自动路由“做题”）
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
                    self.memory.add_assistant(f"✅ 已生成图像：{path}")

                # ---------- 矩阵乘法：矩阵乘法/计算 <A> [*|×|x] <B> ----------
                if tool_obs is None:
                    m2 = re.match(r"^(?:矩阵乘法|计算)\s+(\[.*\])\s*(?:\*|×|x)\s*(\[.*\])\s*$",
                                  txt.replace(" ", ""))
                    if not m2:
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
