import numpy as np
import hnswlib

d, n = 64, 1000
X = np.random.randn(n, d).astype('float32')
ids = np.arange(n, dtype=np.int64)

idx = hnswlib.Index(space='cosine', dim=d)
idx.init_index(max_elements=n, M=16, ef_construction=100)
idx.set_num_threads(1)       # Windows更稳
idx.add_items(X, ids, num_threads=1)
idx.set_ef(100)
print("OK:", idx.get_current_count())
