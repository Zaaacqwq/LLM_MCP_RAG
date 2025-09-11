# agent/retriever.py
import json
import fitz
from pathlib import Path
from agent.embeddings import Embeddings, cosine_sim
from agent.config import RAGConfig


def simple_split(text: str, size: int, overlap: int):
    # 参数合法化
    size = max(1, int(size))
    overlap = max(0, int(overlap))
    if overlap >= size:
        overlap = size - 1

    chunks = []
    n = len(text)
    start = 0
    MAX_CHUNKS = 10000  # 防止死循环

    while start < n and len(chunks) < MAX_CHUNKS:
        end = min(n, start + size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        next_start = start + size - overlap
        if next_start <= start:
            next_start = end
        start = next_start

    return chunks


class Retriever:
    def __init__(self, cfg: RAGConfig, emb: Embeddings):
        self.cfg = cfg
        self.emb = emb
        self.index = {"embeddings": [], "metas": []}

        # 尝试加载已有索引
        try:
            if cfg.index_path.exists() and cfg.index_path.stat().st_size > 0:
                self.index = json.loads(cfg.index_path.read_text("utf-8"))
        except Exception as e:
            print("⚠️ index.json 无效，将重建:", e)
            self.index = {"embeddings": [], "metas": []}

    def build_index_from_docs(self):
        chunks = []
        metas = []
        for p in self.cfg.docs_dir.rglob("*"):
            suffix = p.suffix.lower()
            if suffix in {".txt", ".md"}:
                text = p.read_text("utf-8", errors="ignore")
            elif suffix == ".pdf":
                text = self._read_pdf(p)
            else:
                continue

            parts = simple_split(text, self.cfg.chunk_size, self.cfg.chunk_overlap)
            for i, part in enumerate(parts):
                metas.append({
                    "text": part,
                    "source": str(p),
                    "chunk_id": i
                })
                chunks.append(part)
        return chunks, metas

    def _read_pdf(self, path: Path) -> str:
        """读取 PDF 并返回纯文本"""
        text = []
        with fitz.open(path) as doc:
            for page in doc:
                text.append(page.get_text())
        return "\n".join(text)

    async def ensure_index(self, batch_size: int = 16):
        if self.index["embeddings"]:
            return

        chunks, metas = self.build_index_from_docs()
        if not chunks:
            self.index = {"embeddings": [], "metas": []}
            return

        embeddings, metas_out = [], []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            vecs = await self.emb.embed(batch)
            embeddings.extend(vecs)
            metas_out.extend(metas[i:i + batch_size])

            # 分批落盘，防止中途崩溃
            self.index = {"embeddings": embeddings, "metas": metas_out}
            self.cfg.index_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.cfg.index_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(self.index, ensure_ascii=False), "utf-8")
            tmp.replace(self.cfg.index_path)

    async def topk(self, query: str, k: int):
        await self.ensure_index()
        if not self.index["embeddings"]:
            return []

        qv = (await self.emb.embed([query]))[0]
        scored = []
        for vec, meta in zip(self.index["embeddings"], self.index["metas"]):
            scored.append((cosine_sim(qv, vec), meta))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [meta for _, meta in scored[:k]]
