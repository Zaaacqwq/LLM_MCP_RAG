# agent/chunkers.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Iterable, Any, Dict
import re, json, time, asyncio, os

RAG_DEBUG = os.getenv("RAG_DEBUG", "0") == "1"

def _dbg_print(msg: str):
    if RAG_DEBUG:
        print(f"[RAG-CHUNK] {msg}")

# ========== 基础工具 ==========
def _clean(chunks: List[str]) -> List[str]:
    return [c.strip() for c in chunks if c and c.strip()]

def _greedy_merge(units: Iterable[str], max_len: int, overlap: int) -> List[str]:
    chunks, cur, cur_len = [], [], 0
    for u in units:
        L = len(u)
        if cur_len + L + 1 <= max_len:
            cur.append(u); cur_len += L + 1
        else:
            if cur:
                block = "\n".join(cur).strip()
                if block: chunks.append(block)
                if overlap > 0 and block:
                    tail = block[-overlap:]
                    cur = [tail] if tail.strip() else []
                    cur_len = len(cur[0]) if cur else 0
                else:
                    cur, cur_len = [], 0
            if L > max_len:  # 单元过大硬切
                step = max(1, max_len - overlap)
                for i in range(0, L, step):
                    chunks.append(u[i:i+max_len])
                cur, cur_len = [], 0
            else:
                cur, cur_len = [u], L
    if cur:
        block = "\n".join(cur).strip()
        if block: chunks.append(block)
    return _clean(chunks)

# ========== 1) 传统切割 ==========
class BaseChunker:
    def split(self, text: str) -> List[str]:
        raise NotImplementedError

@dataclass
class FixedChunker(BaseChunker):
    chunk_size: int = 1000
    chunk_overlap: int = 100
    def split(self, text: str) -> List[str]:
        out, i, step = [], 0, max(1, self.chunk_size - self.chunk_overlap)
        while i < len(text):
            out.append(text[i:i+self.chunk_size])
            i += step
        return _clean(out)

@dataclass
class RecursiveChunker(BaseChunker):
    chunk_size: int = 1200
    chunk_overlap: int = 150
    separators: Tuple[str, ...] = (
        r"\n#{1,6} .*?\n",
        r"\n\n+",
        r"\n",
        r"(?<=[。.!?])\s+",
        r"(?<=[，,;；])\s+",
    )
    def split(self, text: str) -> List[str]:
        def _split_by_regex(t: str, pat: str) -> List[str]:
            parts = re.split(pat, t, flags=re.MULTILINE)
            return [p for p in parts if p and p.strip()]
        def _rec(parts: List[str], seps: Tuple[str, ...]) -> List[str]:
            if not seps: return _greedy_merge(parts, self.chunk_size, self.chunk_overlap)
            pat = seps[0]; out=[]
            for p in parts:
                subs = _split_by_regex(p, pat)
                if any(len(s) > self.chunk_size*1.5 for s in subs):
                    out.extend(_rec(subs, seps[1:]))
                else:
                    out.extend(subs)
            return out
        rough = _rec([text], self.separators)
        return _greedy_merge(rough, self.chunk_size, self.chunk_overlap)

@dataclass
class SentenceChunker(BaseChunker):
    chunk_size: int = 1000
    chunk_overlap: int = 100
    SENT = re.compile(r'(?<=[。！？!?.])\s+|(?<=\n)')
    def split(self, text: str) -> List[str]:
        sents = [s.strip() for s in self.SENT.split(text) if s.strip()]
        return _greedy_merge(sents, self.chunk_size, self.chunk_overlap)

@dataclass
class HeadingAwareChunker(BaseChunker):
    chunk_size: int = 1400
    chunk_overlap: int = 150
    def split(self, text: str) -> List[str]:
        blocks = re.split(r'\n(?=#+\s)|\n(?=\d+\.\s)|\n(?=[A-Z][^\n]{0,60}\n[-=]{3,})', text)
        paras = []
        for b in blocks:
            paras.extend(re.split(r'\n{2,}', b.strip()))
        return _greedy_merge([p for p in paras if p.strip()], self.chunk_size, self.chunk_overlap)

# ========== 2) 纯语义切 ==========
@dataclass
class SemanticBoundaryChunker(BaseChunker):
    chunk_size: int = 1200
    chunk_overlap: int = 150
    model_name: str = "all-MiniLM-L6-v2"
    sim_threshold: float = 0.55
    smooth_k: int = 2
    min_sent_per_chunk: int = 3
    max_sent_per_chunk: int = 28
    def __post_init__(self):
        try:
            from sentence_transformers import SentenceTransformer
            self._encode = SentenceTransformer(self.model_name).encode
            self._ok = True
        except Exception:
            self._encode = None; self._ok = False
    def split(self, text: str) -> List[str]:
        sents = [s.strip() for s in re.split(r'(?<=[。！？!?.])\s+|\n+', text) if s.strip()]
        if len(sents) <= self.min_sent_per_chunk: return [text.strip()]
        if not self._ok:
            return _greedy_merge(sents, self.chunk_size, self.chunk_overlap)
        import numpy as np
        emb = self._encode(sents, normalize_embeddings=True)
        sims = np.sum(emb[1:]*emb[:-1], axis=1)
        if self.smooth_k>0:
            k=self.smooth_k; kernel=np.ones(2*k+1)/(2*k+1); sims=np.convolve(sims,kernel,mode="same")
        valleys=set()
        for i in range(1,len(sims)-1):
            if sims[i]<sims[i-1] and sims[i]<sims[i+1] and sims[i]<self.sim_threshold:
                valleys.add(i)
        chunks, cur = [], []
        for i,s in enumerate(sents):
            cur.append(s); cur_len=sum(len(x) for x in cur)
            cut = (i in valleys and len(cur)>=self.min_sent_per_chunk) \
                  or cur_len>=int(self.chunk_size*1.2) \
                  or len(cur)>=self.max_sent_per_chunk
            if cut:
                block="\n".join(cur).strip()
                if block: chunks.append(block)
                cur = [block[-self.chunk_overlap:]] if self.chunk_overlap>0 and block else []
        if cur:
            block="\n".join(cur).strip()
            if block: chunks.append(block)
        return chunks

# ========== 3) LLM 大纲 + 语义细切 ==========
def _outline_prompt(text: str) -> str:
    return (
        "You are a document structuring assistant.\n"
        "Return strictly valid JSON:\n"
        "{ \"sections\": [ {\"title\": string, \"start_char\": int, \"end_char\": int} ] }\n"
        "Rules:\n"
        "- Use 0-based char indices on RAW text.\n"
        "- Keep sections contiguous; avoid gaps/overlaps.\n"
        "- Align to natural units (headings, lists, code fences, tables, figure captions).\n"
        "- Prefer 5–60 sections for long docs; fewer for short.\n\n"
        "TEXT:\n" + text
    )

@dataclass
class LLMOutlineChunker(BaseChunker):
    llm: Any
    chunk_size: int = 1200
    chunk_overlap: int = 150
    max_chars_per_call: int = 18000
    retries: int = 2
    retry_sleep: float = 0.8
    semantic_model: str = "all-MiniLM-L6-v2"
    sim_threshold: float = 0.55

    def __post_init__(self):
        self.semantic = SemanticBoundaryChunker(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            model_name=self.semantic_model,
            sim_threshold=self.sim_threshold
        )

    def _llm_infer(self, prompt: str) -> str:
        messages = [
            {"role":"system","content":"You return strictly valid, concise JSON."},
            {"role":"user","content":prompt},
        ]
        # 优先：同步调用，避免 "coroutine was never awaited"
        if hasattr(self.llm, "complete_sync"):
            return self.llm.complete_sync(messages, temperature=0.0)

        # 兜底：仍然保留异步路径（很少用到）
        async def _call():
            return await self.llm.complete(messages, temperature=0.0)
        try:
            import asyncio
            return asyncio.run(_call())
        except RuntimeError:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(_call())

    def _ask_outline(self, raw: str) -> Dict[str, Any]:
        last=None
        for _ in range(self.retries+1):
            try:
                out = self._llm_infer(_outline_prompt(raw[:self.max_chars_per_call])).strip()
                out = out.strip("` \n")
                if out.lower().startswith("json"): out = out[4:].strip()
                data = json.loads(out)
                if isinstance(data, dict) and "sections" in data: return data
            except Exception as e:
                last=e; time.sleep(self.retry_sleep)
        return {"sections": [], "error": str(last) if last else None}

    def _apply_sections(self, text: str, sections: List[Dict[str,int]]) -> List[str]:
        n=len(text)
        spans=[]
        for s in sections:
            try:
                a=max(0,int(s.get("start_char",0))); b=min(n,int(s.get("end_char",n)))
                if a<b: spans.append((a,b))
            except: pass
        spans.sort(key=lambda x:x[0])
        if not spans: return self.semantic.split(text)
        chunks=[]
        for a,b in spans:
            seg=text[a:b].strip()
            if seg: chunks.extend(self.semantic.split(seg))
        return chunks

    def split(self, text: str) -> List[str]:
        text = text if isinstance(text,str) else str(text)
        if len(text) <= self.max_chars_per_call:
            outline = self._ask_outline(text)
            return self._apply_sections(text, outline.get("sections", []))

        win=self.max_chars_per_call; ov=int(win*0.08)
        starts=list(range(0,len(text), win-ov))
        global_secs=[]
        for s in starts:
            sub=text[s:s+win]
            outline=self._ask_outline(sub)
            for sec in outline.get("sections", []):
                try:
                    a=int(sec["start_char"])+s; b=int(sec["end_char"])+s
                    if 0<=a<b<=len(text): global_secs.append({"start_char":a,"end_char":b})
                except: pass
        global_secs.sort(key=lambda d:d["start_char"])
        merged=[]
        for sec in global_secs:
            if not merged: merged.append(sec); continue
            prev=merged[-1]
            if sec["start_char"] <= prev["end_char"] + 20:
                prev["end_char"] = max(prev["end_char"], sec["end_char"])
            else:
                merged.append(sec)
        return self._apply_sections(text, merged)

# ========== 4) 工厂 ==========
def _wrap_debug(name: str, chunker):
    if not RAG_DEBUG:
        return chunker
    orig_split = chunker.split
    def wrapped(text: str):
        _dbg_print(f"Using chunker={name}, type={chunker.__class__.__name__}")
        t0 = time.time()
        chunks = orig_split(text)
        dur = (time.time()-t0)*1000
        avg = (sum(len(c) for c in chunks)//max(1,len(chunks))) if chunks else 0
        _dbg_print(f"chunks={len(chunks)}, avg_len={avg}, time_ms={int(dur)}")
        if chunks:
            preview = chunks[0][:120].replace("\n"," ")
            _dbg_print(f"preview='{preview}...'")
        return chunks
    chunker.split = wrapped
    return chunker

def get_chunker(
    name: str,
    chunk_size: int = 1200,
    chunk_overlap: int = 150,
    semantic_model: str = "all-MiniLM-L6-v2",
    sim_threshold: float = 0.55,
    llm: Any = None,
    **kwargs
) -> BaseChunker:
    name = (name or "").lower().strip()
    chunk_size = max(1, int(chunk_size))
    chunk_overlap = max(0, int(chunk_overlap))
    if chunk_overlap >= chunk_size:
        chunk_overlap = max(0, chunk_size - 1)

    if name in ("fixed", "char"):
        return _wrap_debug(name, FixedChunker(chunk_size, chunk_overlap))
    if name in ("recursive", "rec"):
        return _wrap_debug(name, RecursiveChunker(chunk_size, chunk_overlap))
    if name in ("sentence", "sent"):
        return _wrap_debug(name, SentenceChunker(chunk_size, chunk_overlap))
    if name in ("heading", "heading_aware"):
        return _wrap_debug(name, HeadingAwareChunker(chunk_size, chunk_overlap))
    if name in ("semantic_ai", "semantic"):
        smooth_k = int(kwargs.get("smooth_k", 2))
        min_sent = int(kwargs.get("min_sent_per_chunk", 3))
        max_sent = int(kwargs.get("max_sent_per_chunk", 28))
        return _wrap_debug(
            name,
            SemanticBoundaryChunker(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                model_name=semantic_model,
                sim_threshold=float(sim_threshold),
                smooth_k=smooth_k,
                min_sent_per_chunk=min_sent,
                max_sent_per_chunk=max_sent,
            )
        )
    if name in ("llm_outline", "llm"):
        if llm is None:
            raise ValueError("LLMOutlineChunker requires `llm` instance.")
        max_chars = int(kwargs.get("max_chars_per_call", 18000))
        return _wrap_debug(
            name,
            LLMOutlineChunker(
                llm=llm,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                max_chars_per_call=max_chars,
                semantic_model=semantic_model,
                sim_threshold=float(sim_threshold),
            )
        )
    supported = [
        "fixed", "recursive", "sentence", "heading",
        "semantic_ai", "semantic", "llm_outline", "llm"
    ]
    raise ValueError(f"Unknown chunker name: {name}. Supported: {supported}")
