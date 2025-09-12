# agent/orchestrator.py
import asyncio
import re
import json
import ast
from typing import List, Dict, Any

from agent.memory import Memory
from agent.retriever import Retriever
from agent.llm import LLM
from agent.config import APP, CHUNK, LLMConfig  # 读取 chunker 配置
from agent.chunkers import get_chunker  # 本地切割工厂

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

        # 预先构造 chunker（仅 llm_outline 需要 llm）
        self._chunker = get_chunker(
            name=CHUNK.name,
            chunk_size=CHUNK.chunk_size,
            chunk_overlap=CHUNK.chunk_overlap,
            semantic_model=CHUNK.semantic_model,
            sim_threshold=CHUNK.semantic_sim_threshold,
            llm=self.llm,
            max_chars_per_call=CHUNK.max_chars_per_call,
        )

    async def _mcp_read_dir(self, docs_dir: str) -> List[Dict[str, Any]]:
        """只通过 MCP 读取文件（file.read_dir）。"""
        if not self.router:
            return []
        args = {
            "dir": docs_dir,
            "patterns": ["*.pdf", "*.txt", "*.md", "*.docx"],
            "recursive": True,
            "limit": None,
            "normalize": True,
        }
        try:
            res = await _tools_call(self.router, "file.read_dir", args)
            return res.get("files", []) if isinstance(res, dict) else []
        except Exception:
            # MCP 不可用或失败就返回空，外面会做本地兜底
            return []

    async def reindex(self) -> dict:
        """
        重新构建索引：
          1) 用 MCP 读取原始文件内容（仅 read_dir）
          2) 用本地 chunker 切割（可走 LLM 大纲）
          3) 写入 retriever（带 built_by 和 docs_sig）
        """
        print("强制重建索引中...")

        # 1) 读取文件（优先 MCP）
        files = await self._mcp_read_dir(str(APP.rag.docs_dir))
        used_mcp = bool(files)

        # 2) 本地兜底读取（若 MCP 不可用）
        if not files:
            from pathlib import Path
            print("➡️ 使用本地文件读取 fallback")
            base = Path(APP.rag.docs_dir)
            for p in base.rglob("*"):
                if p.is_file() and p.suffix.lower() in (".pdf", ".txt", ".md", ".docx"):
                    try:
                        content = p.read_text("utf-8", errors="ignore")
                    except Exception:
                        content = ""
                    try:
                        st = p.stat()
                        mtime = int(st.st_mtime)
                        size = int(st.st_size)
                    except Exception:
                        mtime = 0
                        size = len(content)
                    files.append({
                        "meta": {"source": str(p), "mtime": mtime, "size": size},
                        "content": content
                    })

        # 2.5) 计算 docs 签名（统一：按 source/mtime/size）
        import hashlib
        sig_items = []
        for f in files:
            meta = f.get("meta") or {}
            src = str(meta.get("source") or meta.get("path") or "unknown")
            mtime = int(meta.get("mtime", 0)) if isinstance(meta.get("mtime", 0), (int, float)) else 0
            size_meta = int(meta.get("size", 0)) if isinstance(meta.get("size", 0), (int, float)) else 0
            content = f.get("content") or ""
            size_for_sig = size_meta if size_meta > 0 else len(content)
            sig_items.append((src, mtime, int(size_for_sig)))
        h = hashlib.sha256()
        for k, m, s in sorted(sig_items, key=lambda x: x[0]):
            h.update(str(k).encode());
            h.update(b"|")
            h.update(str(int(m)).encode());
            h.update(b"|")
            h.update(str(int(s)).encode());
            h.update(b"\n")
        docs_sig = h.hexdigest()

        # 3) 本地切割（替代 MCP file.chunk），chunker 名称来自 config
        texts, metadatas = [], []
        total_chunks = 0
        for f in files:
            text = (f.get("content") or "").strip()
            if not text:
                continue
            meta = f.get("meta") or {}
            chunks = self._chunker.split(text)
            total_chunks += len(chunks)
            texts.extend(chunks)
            metadatas.extend([meta | {"chunk_id": i} for i in range(len(chunks))])

        # 4) 写入索引（尽量适配不同 Retriever 实现）
        async def maybe_await(x):
            return await x if asyncio.iscoroutine(x) else x

        try:
            built_by = f"reader:{'mcp' if used_mcp else 'local'}; chunker:{CHUNK.name}"
            if hasattr(self.retriever, "rebuild_from_texts"):
                await maybe_await(self.retriever.rebuild_from_texts(
                    texts, metadatas, built_by=built_by, docs_sig=docs_sig
                ))
            elif hasattr(self.retriever, "rebuild"):
                # 旧接口：无法传 built_by/docs_sig，只能维持旧行为
                await maybe_await(self.retriever.rebuild(texts=texts, metadatas=metadatas))
            elif hasattr(self.retriever, "upsert_many"):
                await maybe_await(self.retriever.upsert_many(texts, metadatas, rebuild=True))
            elif hasattr(self.retriever, "add_texts"):
                await maybe_await(self.retriever.add_texts(texts, metadatas))
            else:
                return {"chunks": total_chunks, "built_by": built_by + "; no-op"}
        except Exception as e:
            return {"chunks": total_chunks, "built_by": f"error:{e}"}

        return {"chunks": total_chunks, "built_by": built_by}

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

                # ---------- 矩阵乘法 ----------
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

                # ---------- 求导 ----------
                if tool_obs is None:
                    m3 = re.match(r"^对\s+(.+?)\s+求导\s*$", txt)
                    if m3:
                        expr = m3.group(1)
                        res = await _tools_call(self.router, "math.diff", {"expr": expr, "var": "x"})
                        tool_obs = f"[math.diff] {json.dumps(res, ensure_ascii=False)}"
                        self.memory.add_assistant(f"∂/∂x {expr} = {res.get('derivative')}")

                # ---------- 积分 ----------
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
