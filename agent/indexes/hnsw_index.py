# agent/indexes/hnsw_index.py
from __future__ import annotations
from typing import Tuple
from pathlib import Path
import numpy as np

# ===== Flat 兜底索引（纯 NumPy 余弦）=====
class FlatIndex:
    kind = "flat"
    def __init__(self, dim: int):
        self.dim = int(dim)
        self.X: np.ndarray | None = None
        self.ids: np.ndarray | None = None

    @staticmethod
    def _cosine_knn(Q: np.ndarray, X: np.ndarray, k: int):
        Qn = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-12)
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        S = Qn @ Xn.T
        idx = np.argpartition(-S, kth=min(k-1, S.shape[1]-1), axis=1)[:, :k]
        row = np.arange(Q.shape[0])[:, None]
        sims_topk = S[row, idx]
        order = np.argsort(-sims_topk, axis=1)
        idx_sorted = idx[row, order]
        sims_sorted = sims_topk[row, order]
        dists = 1.0 - sims_sorted  # 与 hnsw 对齐：返回距离
        return idx_sorted, dists

    def add_batch(self, X: np.ndarray, ids: np.ndarray) -> None:
        self.X = X.astype(np.float32, copy=False)
        self.ids = ids.astype(np.int64, copy=False)

    def search(self, Q: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        assert self.X is not None and self.ids is not None
        local_idx, dists = self._cosine_knn(Q.astype(np.float32), self.X, int(k))
        labels = self.ids[local_idx]
        return labels, dists

    def save(self, path_prefix: Path) -> None:
        np.savez_compressed(path_prefix.with_suffix(".npz"), X=self.X, ids=self.ids)

    def load(self, path_prefix: Path) -> None:
        data = np.load(path_prefix.with_suffix(".npz"))
        self.X = data["X"].astype(np.float32, copy=False)
        self.ids = data["ids"].astype(np.int64, copy=False)
        self.dim = int(self.X.shape[1])


# ===== HNSW 索引（hnswlib）=====
class HNSWIndex:
    kind = "hnsw"
    def __init__(self, dim: int, space: str = "cosine", M: int = 32, ef_construction: int = 200, ef_search: int = 120):
        try:
            import hnswlib
        except Exception as e:
            raise RuntimeError(f"hnswlib not available: {e}")
        self.hnswlib = hnswlib
        self.dim = int(dim)
        self.space = space
        self.M = int(M)
        self.efc = int(ef_construction)
        self.efs = int(ef_search)
        self.index = hnswlib.Index(space=space, dim=self.dim)
        self.initialized = False

    def _ensure(self, cap: int):
        if not self.initialized:
            self.index.init_index(max_elements=int(cap), M=self.M, ef_construction=self.efc)
            self.index.set_num_threads(1)   # Windows 更稳：固定 1 线程
            self.index.set_ef(self.efs)
            self.initialized = True

    @staticmethod
    def _prep_X(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32, order="C")
        # 保证二维 + 维度正确 + 连续内存
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")
        if not X.flags["C_CONTIGUOUS"]:
            X = np.ascontiguousarray(X, dtype=np.float32)
        # 清洗 NaN/Inf（虽然 retriever 已做，双保险）
        if not np.isfinite(X).all():
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        return X

    @staticmethod
    def _prep_ids(ids: np.ndarray) -> np.ndarray:
        ids = np.asarray(ids, dtype=np.int64, order="C")
        if ids.ndim != 1:
            ids = ids.reshape(-1)
        if not ids.flags["C_CONTIGUOUS"]:
            ids = np.ascontiguousarray(ids, dtype=np.int64)
        return ids

    def add_batch(self, X: np.ndarray, ids: np.ndarray) -> None:
        X = self._prep_X(X)
        ids = self._prep_ids(ids)
        if X.shape[1] != self.dim:
            raise ValueError(f"dim mismatch: X has {X.shape[1]} vs index dim {self.dim}")
        if len(ids) != X.shape[0]:
            raise ValueError(f"ids len {len(ids)} != X rows {X.shape[0]}")
        self._ensure(len(ids))
        # 大批量喂入时，分段 add，避免极端情况下触发原生崩溃
        step = 2048 if X.shape[0] > 2048 else X.shape[0]
        for s in range(0, X.shape[0], step):
            e = min(s + step, X.shape[0])
            self.index.add_items(X[s:e], ids[s:e], num_threads=1)

    def search(self, Q: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        Q = np.asarray(Q, dtype=np.float32, order="C")
        if Q.ndim == 1:
            Q = Q.reshape(1, -1)
        if not Q.flags["C_CONTIGUOUS"]:
            Q = np.ascontiguousarray(Q, dtype=np.float32)
        if not np.isfinite(Q).all():
            Q = np.nan_to_num(Q, nan=0.0, posinf=0.0, neginf=0.0)
        labels, dists = self.index.knn_query(Q, k=int(k), num_threads=1)
        return labels, dists

    def save(self, path_prefix: Path) -> None:
        self.index.save_index(str(path_prefix.with_suffix(".bin")))

    def load(self, path_prefix: Path) -> None:
        self.index = self.hnswlib.Index(space=self.space, dim=self.dim)
        self.index.load_index(str(path_prefix.with_suffix(".bin")))
        self.index.set_num_threads(1)   # 同样固定 1 线程
        self.index.set_ef(self.efs)
        self.initialized = True
