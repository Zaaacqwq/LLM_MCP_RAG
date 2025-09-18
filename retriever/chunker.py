# retriever/chunker.py
from __future__ import annotations
import re
from typing import List
from agent.llm import LLM

def simple_chunk(text: str, chunk_size: int = 700, overlap: int = 120) -> List[str]:
    # 你原来的实现（保留作回退）：
    chunks = []
    i = 0
    n = len(text)
    while i < n:
        j = min(i + chunk_size, n)
        chunks.append(text[i:j])
        if j == n: break
        i = j - overlap
        if i < 0: i = 0
    return chunks
# （原实现参见：:contentReference[oaicite:6]{index=6}）

def ai_chunk(
    text: str,
    target_chars: int = 800,
    max_chunk_chars: int = 1200,
    max_chunks: int = 200,
) -> List[str]:
    """
    用小提示词让 LLM 产生语义分段，控制单块长度与最大块数。
    失败或返回异常时回退到 simple_chunk。
    """
    text = text.strip()
    if not text:
        return []
    # 粗预切：按明显段落/标题先断开，避免丢格式
    prelim = re.split(r"\n(?=#+\s|\d+\.\s|-{3,}|={3,}|\s*$)", text)
    prelim = [p.strip() for p in prelim if p.strip()]
    sample = "\n\n---\n\n".join(prelim[:5])  # 给 LLM 一点上下文样例

    prompt = (
        "You are a chunker. Split the DOCUMENT into semantically coherent chunks.\n"
        f"- Aim each chunk around {target_chars} characters; never exceed {max_chunk_chars}.\n"
        f"- Preserve headings and list structure; do not rewrite.\n"
        f"- Return ONLY the chunks, separated by a line with exactly three dashes: ---\n"
        f"- Cap at {max_chunks} chunks total.\n"
        "DOCUMENT:\n"
        f"{text[:12000]}\n\n"
        "Example of the separator:\n---\n"
        "Output:"
    )
    try:
        llm = LLM()
        out = llm.complete_sync(
            [{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=1024,
        )
        # 拆分返回（严格只认独立一行的 ---）
        cand = [c.strip() for c in re.split(r"\n-{3,}\n", out) if c.strip()]
        # 长度兜底，超限再细切
        chunks: List[str] = []
        for c in cand:
            if len(c) <= max_chunk_chars:
                chunks.append(c)
            else:
                chunks.extend(simple_chunk(c, max_chunk_chars, max(0, max_chunk_chars - target_chars)))
        if not chunks:
            raise RuntimeError("empty ai chunks")
        return chunks[:max_chunks]
    except Exception:
        # 回退：简单切块
        return simple_chunk(text, chunk_size=target_chars, overlap=target_chars // 5)
