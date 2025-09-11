import json, re
from pathlib import Path
from .embeddings import Embeddings, cosine_sim
from .config import RAGConfig

def simple_split(text: str, size: int, overlap: int):
    # 1) 参数合法化
    size = max(1, int(size))
    overlap = max(0, int(overlap))
    if overlap >= size:
        overlap = size - 1  # 保证每次都有正向前进

    chunks = []
    n = len(text)
    start = 0

    # 2) 安全上限，防意外死循环
    MAX_CHUNKS = 10000

    while start < n and len(chunks) < MAX_CHUNKS:
        end = min(n, start + size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # 关键修复：一旦到文本末尾，立即退出，避免重复尾块
        if end == n:
            break

        # 正常前进：上一个起点 + size - overlap
        next_start = start + size - overlap

        # 3) 再保险：若没有前进，则强制前进一步
        if next_start <= start:
            next_start = end

        start = next_start

    return chunks

class Retriever:
    def __init__(self, cfg: RAGConfig, emb: Embeddings):
        self.cfg = cfg
        self.emb = emb
        self.index = {"embeddings": [], "metas": []}
        if cfg.index_path.exists():
            self.index = json.loads(cfg.index_path.read_text("utf-8"))

    def build_index_from_docs(self):
        docs = []
        for p in self.cfg.docs_dir.rglob("*"):
            if p.suffix.lower() in {".txt", ".md"}:
                docs.append(p.read_text("utf-8", errors="ignore"))
        chunks = []
        for d in docs:
            chunks.extend(simple_split(d, self.cfg.chunk_size, self.cfg.chunk_overlap))
        return chunks

    # 替换 ensure_index 为分批实现
    async def ensure_index(self, batch_size: int = 16):  # 小批量
        if self.index["embeddings"]:
            return
        chunks = self.build_index_from_docs()
        if not chunks:
            self.index = {"embeddings": [], "metas": []}
            return

        embeddings, metas = [], []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            vecs = await self.emb.embed(batch)
            embeddings.extend(vecs)
            metas.extend([{"text": c} for c in batch])

            # 原子落盘，避免半成品 index 导致下次 JSONDecodeError
            self.index = {"embeddings": embeddings, "metas": metas}
            self.cfg.index_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.cfg.index_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(self.index, ensure_ascii=False), "utf-8")
            tmp.replace(self.cfg.index_path)

    async def topk(self, query: str, k: int):
        await self.ensure_index()
        if not self.index["embeddings"]:
            return []
        qv = (await self.emb.embed([query]))[0]
        scored = [(cosine_sim(qv, vec), meta["text"])
                  for vec, meta in zip(self.index["embeddings"], self.index["metas"])]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [t for _, t in scored[:k]]

