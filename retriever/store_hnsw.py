import hnswlib, numpy as np, os, json

class HNSWStore:
    def __init__(self, dim: int, index_dir: str):
        self.dim = dim
        self.index_dir = index_dir
        os.makedirs(index_dir, exist_ok=True)
        self.idx_path = os.path.join(index_dir, "knn.bin")
        self.meta_path = os.path.join(index_dir, "meta.json")
        self.index = None
        self.meta = []

    def build(self, vectors, metas):
        self.index = hnswlib.Index(space='cosine', dim=self.dim)
        self.index.init_index(max_elements=len(vectors), ef_construction=200, M=16)
        self.index.add_items(np.array(vectors), np.arange(len(vectors)))
        self.index.set_ef(64)
        self.meta = metas
        self.index.save_index(self.idx_path)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.meta, f, ensure_ascii=False)

    def load(self):
        if not os.path.exists(self.idx_path):
            print(f"[hnsw] missing index file: {self.idx_path}")
            return False
        self.index = hnswlib.Index(space='cosine', dim=self.dim)
        self.index.load_index(self.idx_path)
        with open(self.meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)
        self.index.set_ef(64)
        print(f"[hnsw] loaded index: {self.idx_path}, metas={len(self.meta)}")
        return True

    def search(self, vec, top_k=5):
        if self.index is None:
            if not self.load(): return []
        labels, dists = self.index.knn_query(np.array([vec]), k=top_k)
        res = []
        for lab, dist in zip(labels[0], dists[0]):
            meta = self.meta[int(lab)]
            res.append((float(1 - dist), meta))  # cosine sim
        return res
