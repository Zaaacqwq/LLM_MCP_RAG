import json
from pathlib import Path
from .config import RAGConfig
from .embeddings import Embeddings, cosine_sim

class Retriever:
    def __init__(self, cfg: RAGConfig, emb: Embeddings):
        self.cfg = cfg
        self.emb = emb
        self.index = {"embeddings": [], "metas": []}
        try:
            if cfg.index_path.exists() and cfg.index_path.stat().st_size > 0:
                self.index = json.loads(cfg.index_path.read_text("utf-8"))
        except Exception as e:
            # 索引损坏时忽略，后面 ensure_index 会重建
            print("⚠️ index.json invalid, will rebuild:", e)
            self.index = {"embeddings": [], "metas": []}
