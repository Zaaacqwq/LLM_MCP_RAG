# retriever/embedder.py
from __future__ import annotations
import hashlib
import numpy as np
from typing import Iterable

class HashEmbedder:
    """
    轻量、确定性的本地 embedding：
    - 用字符 3-gram 做特征哈希到固定维度
    - 使用±1 累加并做 L2 归一化
    - 相同文本 → 同一向量；越相似的文本 → 统计上越接近
    """
    def __init__(self, dim: int = 384, salt: str = "rag-local"):
        assert dim >= 64
        self.dim = dim
        self.salt = salt.encode("utf-8")

    def _ngrams(self, text: str, n: int = 3) -> Iterable[bytes]:
        t = text.strip()
        if len(t) < n:
            if t:
                yield t.encode("utf-8")
            return
        bs = t.encode("utf-8")
        # 逐字节 3-gram（稳定、跨平台一致）
        for i in range(len(bs) - n + 1):
            yield bs[i:i+n]

    def embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        if not text:
            return vec
        for ng in self._ngrams(text, 3):
            h = hashlib.blake2b(ng, key=self.salt, digest_size=8).digest()
            idx = int.from_bytes(h[:4], "little") % self.dim
            sgn = 1.0 if (h[4] & 1) == 0 else -1.0
            vec[idx] += sgn
        nrm = float(np.linalg.norm(vec))
        if nrm > 0:
            vec /= nrm
        return vec
