from typing import List
import json, re  # 新增：用于 /matmul 与 /plot 参数解析
from .schema import OrchestratorResult, Message, RetrieveHit, ToolCall
from retriever.pipeline import query as retrieve_query
from tools.tool_router import call as tool_call
from config.settings import settings
from agent.llm import LLM

# ====== helpers ======
def _unescape_one_line(code: str) -> str:
    """
    仅把未转义的 \\n / \\t 展开成真正换行/制表，
    保留字符串中的 \\n（双反斜杠）不动。
    例：'line1\\nline2' -> 'line1\nline2'
        'printf(\"Hello C!\\\\n\");' 保持不变
    """
    code = re.sub(r'(?<!\\)\\n', '\n', code)
    code = re.sub(r'(?<!\\)\\t', '\t', code)
    return code

def _strip_cmd(text: str, cmd: str) -> str:
    return text.replace(cmd, "", 1).strip()

def _parse_matmul_args(s: str):
    """
    支持：
      1) /matmul [[1,2],[3,4]] * [[5,6],[7,8]]
      2) /matmul A=[[1,2,3],[4,5,6]] B=[[7,8],[9,10],[11,12]]
    """
    s = s.strip()
    # 形式1：矩阵 * 矩阵（或 x、×）
    m = re.match(r'^\s*(\[[\s\S]+\])\s*([*x×])\s*(\[[\s\S]+\])\s*$', s, flags=re.I)
    if m:
        A = json.loads(m.group(1))
        B = json.loads(m.group(3))
        return A, B
    # 形式2：A=..., B=...
    mA = re.search(r'A\s*=\s*(\[[\s\S]+\])', s)
    mB = re.search(r'B\s*=\s*(\[[\s\S]+\])', s)
    if mA and mB:
        A = json.loads(mA.group(1))
        B = json.loads(mB.group(1))
        return A, B
    raise ValueError("矩阵输入格式不正确。示例：/matmul [[1,2],[3,4]] * [[5,6],[7,8]]")

def _parse_plot_args(s: str):
    """
    支持：
      /plot sin(x)/x
      /plot sin(x)/x [-5, 5]
      /plot sin(x)/x -5 5
    """
    s = s.strip()
    # [a,b]
    m = re.match(r'^(.*)\[\s*([+-]?\d+\.?\d*)\s*,\s*([+-]?\d+\.?\d*)\s*\]\s*$', s)
    if m:
        return m.group(1).strip(), float(m.group(2)), float(m.group(3))
    # 尾部两个数字
    m = re.match(r'^(.*)\s([+-]?\d+\.?\d*)\s([+-]?\d+\.?\d*)\s*$', s)
    if m:
        return m.group(1).strip(), float(m.group(2)), float(m.group(3))
    # 只给表达式：默认区间
    return s, -10.0, 10.0


class Orchestrator:
    def __init__(self, memory):
        self.memory = memory
        self._llm = None

    async def step(self, text: str, mode: str | None = None) -> OrchestratorResult:
        mode = mode or self._infer_mode(text)
        used_tools: List[ToolCall] = []
        retrieve_hits: List[RetrieveHit] = []
        errors: List[str] = []

        try:
            if mode == "solve":
                expr = _strip_cmd(text, "/solve")
                res = tool_call("math.solve_equation", {"expr": expr, "var": "x"})
                used_tools.append(ToolCall(name="math.solve_equation", args={"expr": expr, "var": "x"}, via="mcp"))
                final = f"答案：{res.get('solutions')}"

            elif mode == "diff":
                expr = _strip_cmd(text, "/diff")
                res = tool_call("math.diff", {"expr": expr, "var": "x"})
                used_tools.append(ToolCall(name="math.diff", args={"expr": expr, "var": "x"}, via="mcp"))
                final = f"导数：{res.get('derivative')}"

            elif mode == "integrate":
                expr = _strip_cmd(text, "/integrate")
                res = tool_call("math.integrate", {"expr": expr, "var": "x"})
                used_tools.append(ToolCall(name="math.integrate", args={"expr": expr, "var": "x"}, via="mcp"))
                final = f"不定积分：{res.get('integral')} + C"

            elif mode == "matmul":
                body = _strip_cmd(text, "/matmul")
                A, B = _parse_matmul_args(body)
                res = tool_call("math.matrix_multiply", {"A": A, "B": B})
                used_tools.append(ToolCall(name="math.matrix_multiply", args={"A": A, "B": B}, via="mcp"))
                final = f"结果矩阵：{res.get('matrix')}，形状：{res.get('shape')}"

            elif mode == "plot":
                body = _strip_cmd(text, "/plot")
                expr, start, end = _parse_plot_args(body)
                res = tool_call("math.plot", {"expr": expr, "var": "x", "start": start, "end": end})
                used_tools.append(ToolCall(name="math.plot", args={"expr": expr, "start": start, "end": end}, via="mcp"))
                final = f"已生成：{res.get('path')}（{res.get('width_px')}×{res.get('height_px')} px, 区间 {res.get('x_range')}，采样 {res.get('samples')}）"



            elif mode == "run":
                # 期望格式（单行）：
                # /run python print(sum(i for i in range(10)))
                # /run c    #include <stdio.h>\nint main(){printf("Hello C!\\n");return 0;}
                # /run cpp  #include <iostream>\nusing namespace std;\nint main(){cout<<"Hello C++!"<<endl;return 0;}
                # /run java public class Main{public static void main(String[] args){System.out.println("Hello Java!");}}
                parts = text.split(maxsplit=2)
                if len(parts) < 3:
                    raise ValueError("用法：/run <python|c|cpp|c++|cxx|java> <code-单行，可含\\n>")
                _, lang_raw, code_raw = parts
                lang = lang_raw.lower()

                # 把未转义的 \n/\t 展开成真正换行（字符串里的 \\n 不动）
                code = _unescape_one_line(code_raw)

                if not code.strip():
                    raise ValueError("code 不能为空。请把代码与命令放在同一行；多行请用 \\n 连接。")

                lang2tool = {
                    "python": "python.run_code", "py": "python.run_code",
                    "c": "c.run_code",
                    "cpp": "cpp.run_code", "c++": "cpp.run_code", "cxx": "cpp.run_code",
                    "java": "java.run_code",
                }
                tool_name = lang2tool.get(lang)
                if not tool_name:
                    raise ValueError(f"不支持的语言：{lang_raw}")

                res = tool_call(tool_name, {"code": code, "timeout": float(getattr(settings, "RUNNER_TIMEOUT_SEC", 3.0))})
                used_tools.append(ToolCall(name=tool_name, args={"timeout": getattr(settings, "RUNNER_TIMEOUT_SEC", 3.0)}, via="mcp"))

                if tool_name == "python.run_code":
                    final = f"stdout:\n{res.get('stdout', '')}\n\nstderr:\n{res.get('stderr', '')}"
                else:
                    comp = res.get("compile", {})
                    run = res.get("run", {})
                    final = (
                        "=== compile ===\n"
                        f"exit: {comp.get('exit_code')}  time: {comp.get('time_ms')}ms\n"
                        f"stdout:\n{comp.get('stdout', '')}\n"
                        f"stderr:\n{comp.get('stderr', '')}\n\n"
                        "=== run ===\n"
                        f"exit: {run.get('exit_code')}  time: {run.get('time_ms', '')}ms\n"
                        f"stdout:\n{run.get('stdout', '')}\n"
                        f"stderr:\n{run.get('stderr', '')}\n"
                    )

            elif mode == "reindex":
                from retriever.indexer import reindex
                stats = reindex()
                final = f"✅ 索引已重建  (chunks={stats['chunks']})"

            else:
                import re
                from typing import List

                # —— 通用工具 —— #
                def tokenize(s: str) -> List[str]:
                    return [t for t in re.split(r"[^0-9a-zA-Z]+", s.lower()) if t]

                def build_subqueries(q: str) -> List[str]:
                    """
                    通用多子查询：原始查询 + 清洗版 + 数字/短语切片
                    - 适配任何主题；无课程/文件名硬编码
                    """
                    qs = [q]
                    ql = q.lower().strip()
                    # 轻清洗（中英混排的常见字符替换）
                    zh = q.replace("：", ":").replace("（", "(").replace("）", ")")
                    if zh != q: qs.append(zh)
                    # 拆短语（防止长句导致语义稀释）
                    toks = tokenize(q)
                    if len(toks) >= 2:
                        qs.append(" ".join(toks[:4]))
                    # 数字片段（如 “2”, “02”, 年份/编号等都能泛化）
                    nums = [t for t in toks if t.isdigit()]
                    for n in nums:
                        qs += [n, f"{int(n):02d}"]
                    # 去重
                    seen, out = set(), []
                    for s in qs:
                        s = s.strip()
                        if s and s.lower() not in seen:
                            seen.add(s.lower());
                            out.append(s)
                    return out

                def overlap_score(query: str, text: str, alpha: float = 1.0) -> float:
                    """
                    通用重合度：query tokens 与 text tokens 的对称归一化重叠
                    """
                    q = set(tokenize(query));
                    t = set(tokenize(text))
                    if not q or not t: return 0.0
                    inter = len(q & t)
                    return alpha * (inter / ((len(q) * len(t)) ** 0.5))

                def path_bonus(query: str, path: str) -> float:
                    """
                    路径词重合（目录名/文件名对任何语料都适用；无硬编码）
                    """
                    segs = [p for p in re.split(r"[\\/]+", path.lower()) if p]
                    return max((overlap_score(query, s, 0.12) for s in segs), default=0.0)

                def title_bonus(query: str, chunk_text: str) -> float:
                    """
                    标题/开头重合：取 chunk 首两行，适配 lecture/报告/合同/博客等任意文体
                    """
                    head = "\n".join(chunk_text.strip().splitlines()[:2])
                    return overlap_score(query, head, 0.10)

                def keyword_bonus(query: str, chunk_text: str) -> float:
                    """
                    关键词计数：对 >=3 字符的查询 token 做出现次数统计（通用）
                    """
                    qts = set(t for t in tokenize(query) if len(t) >= 3)
                    if not qts: return 0.0
                    t = chunk_text.lower()
                    hits = sum(t.count(k) for k in qts)
                    return 0.03 * hits

                # —— 多路召回 —— #
                subqs = build_subqueries(text)
                print(f"[retriever] subqueries={subqs}")

                seen_ids = set()
                merged = []
                for sq in subqs:
                    hits = retrieve_query(sq, settings.TOP_K)  # -> [(dense_score, meta), ...]
                    for dense, meta in hits:
                        hid = f"{meta['path']}#{meta['chunk_id']}"
                        if hid in seen_ids: continue
                        seen_ids.add(hid)
                        merged.append(RetrieveHit(
                            doc_id=hid,
                            score=float(dense),  # 先存向量相似度
                            chunk=meta['text'],  # 你当前存的是前200字符，可考虑存更长前缀
                            source=meta['path'],
                        ))

                # —— 通用融合打分（无任何语料/课程硬编码）—— #
                W_DENSE, W_PATH, W_TITLE, W_KW = 1.00, 0.40, 0.30, 0.60
                for h in merged:
                    h.score = (
                            W_DENSE * h.score
                            + W_PATH * path_bonus(text, h.source)
                            + W_TITLE * title_bonus(text, h.chunk)
                            + W_KW * keyword_bonus(text, h.chunk)
                    )

                merged.sort(key=lambda x: x.score, reverse=True)
                top_hits = merged[:max(8, settings.TOP_K)]

                print(f"[retriever] merged_hits={len(merged)}, top_hits={len(top_hits)}")
                for i, h in enumerate(top_hits[:5]):
                    print(f"[retriever] hit{i}: score={h.score:.3f} path={h.source}")

                ctx_snips = [f"[source: {h.source}, score={h.score:.3f}]\n{h.chunk}" for h in top_hits]
                ctx_text = "\n\n".join(ctx_snips) if ctx_snips else "(none)"

                llm = self._llm or LLM();
                self._llm = llm
                system_prompt = (
                    "You are a concise, precise assistant. Use the provided CONTEXT when relevant. "
                    "If the context is not relevant or empty, answer from your general knowledge. "
                    "Do not fabricate citations."
                )
                user_prompt = f"QUESTION:\n{text}\n\nCONTEXT:\n{ctx_text}"

                final = llm.complete_sync(
                    [{"role": "system", "content": system_prompt},
                     {"role": "user", "content": user_prompt}],
                    temperature=0.2
                )



        except Exception as e:
            errors.append(str(e))
            final = f"发生错误：{e!r}"

        return OrchestratorResult(final_text=final, used_tools=used_tools, retrieve_hits=retrieve_hits, errors=errors)

    def _infer_mode(self, text: str) -> str:
        if text.startswith("/solve"): return "solve"
        if text.startswith("/diff"): return "diff"
        if text.startswith("/integrate"): return "integrate"
        if text.startswith("/matmul"): return "matmul"
        if text.startswith("/plot"): return "plot"
        if text.startswith("/run"): return "run"
        if text.startswith("/reindex"): return "reindex"
        if text.startswith("/explain"): return "explain"
        return "qa"
