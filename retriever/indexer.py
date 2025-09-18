# retriever/indexer.py
from __future__ import annotations
from typing import List, Dict
from config.settings import settings
from .chunker import ai_chunk, simple_chunk
from .embedder import HashEmbedder
from .store_hnsw import HNSWStore
from .reader import read_dir_via_mcp
import os

def reindex(docs_dir: str | None = None, index_dir: str | None = None):
    docs_dir = docs_dir or settings.DOCS_DIR
    index_dir = index_dir or settings.INDEX_DIR

    print(f"[reindex] DOCS_DIR={docs_dir}")
    print(f"[reindex] INDEX_DIR={index_dir} (abs={os.path.abspath(index_dir)})")

    files = read_dir_via_mcp(docs_dir)
    print(f"[reindex] files={len(files)}")
    for f in files[:3]:
        print(f"[reindex] sample: {f['path']} len(text)={len(f['text'])}")

    embedder = HashEmbedder(dim=settings.EMBED_DIM, salt="rag-local")
    vectors, metas = [], []
    for f in files:
        path, text = f["path"], f["text"]
        chunks = ai_chunk(
            text,
            target_chars=getattr(settings, "CHUNK_SIZE", 700),
            max_chunk_chars=getattr(settings, "CHUNK_SIZE", 700) + 400,
        ) or simple_chunk(text, getattr(settings, "CHUNK_SIZE", 700), getattr(settings, "CHUNK_OVERLAP", 120))
        for i, ch in enumerate(chunks):
            vec = embedder.embed(ch)
            vectors.append(vec)
            metas.append({"path": path, "chunk_id": i, "text": ch[:200]})

    store = HNSWStore(dim=settings.EMBED_DIM, index_dir=index_dir)
    if vectors:
        store.build(vectors, metas)
    print(f"[reindex] done: chunks={len(vectors)}")
    return {"files": len(files), "chunks": len(vectors)}

def query(q: str, top_k: int | None = None):
    embedder = HashEmbedder(dim=settings.EMBED_DIM, salt="rag-local")  # ← 改这里
    vec = embedder.embed(q)
    store = HNSWStore(dim=settings.EMBED_DIM, index_dir=settings.INDEX_DIR)
    print(f"[query] INDEX_DIR={settings.INDEX_DIR}")
    hits = store.search(vec, top_k or settings.TOP_K)
    return hits
