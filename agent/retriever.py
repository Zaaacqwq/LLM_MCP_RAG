# agent/retriever.py
from __future__ import annotations
import json, os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np

from agent.config import RAGConfig, CHUNK, LLMConfig
from agent.chunkers import get_chunker
from agent.llm import LLM
from agent.embeddings import Embeddings

# 新索引后端
from agent.indexes.vector_index import VectorIndex
from agent.indexes.hnsw_index import HNSWIndex, FlatIndex


def _hash_list(items: List[Tuple[str, int, int]]) -> str:
    import hashlib
    h = hashlib.sha256()
    for k, m, s in sorted(items, key=lambda x: x[0]):
        h.update(str(k).encode("utf-8")); h.update(b"|")
        h.update(str(int(m)).encode("utf-8")); h.update(b"|")
        h.update(str(int(s)).encode("utf-8")); h.update(b"\n")
    return h.hexdigest()


class Retriever:
    """
    读取 → 切块 → 嵌入 → 索引（HNSW 优先，Flat 兜底）
    - 元数据：index.json
    - 向量索引：.bin (HNSW) / .npz (Flat)
    - built_by: "reader:mcp; chunker:xxx" / "reader:local; chunker:xxx"
    """
    # HNSW 缺省
    _HNSW_SPACE = "cosine"
    _HNSW_M = 32
    _HNSW_EFC = 200
    _HNSW_EFS = 120

    def __init__(self, cfg: RAGConfig, emb: Embeddings):
        self.cfg = cfg
        self.emb = emb
        self.router = None  # CLI 注入
        self.index: Dict[str, Any] = {"metas": [], "built_by": "", "docs_sig": "", "hnsw": {}}

        # 持久化路径
        self._json_path: Path = self.cfg.index_path
        self._idx_prefix: Path = self.cfg.index_path  # 后端决定后缀

        # 组件
        self._llm = LLM(LLMConfig())
        self._chunker = get_chunker(
            name=CHUNK.name,
            chunk_size=CHUNK.chunk_size,
            chunk_overlap=CHUNK.chunk_overlap,
            semantic_model=CHUNK.semantic_model,
            sim_threshold=CHUNK.semantic_sim_threshold,
            llm=self._llm,
            max_chars_per_call=CHUNK.max_chars_per_call,
        )

        # 读 index.json
        try:
            if self._json_path.exists() and self._json_path.stat().st_size > 0:
                self.index = json.loads(self._json_path.read_text("utf-8"))
        except Exception as e:
            print("⚠️ index.json 读取失败，将重建：", e)
            self.index = {"metas": [], "built_by": "", "docs_sig": "", "hnsw": {}}

        # 维度
        self._dim: Optional[int] = None
        try:
            self._dim = int(self.index.get("hnsw", {}).get("dim")) if isinstance(self.index.get("hnsw"), dict) else None
        except Exception:
            self._dim = None

        # 后端
        self._backend: Optional[VectorIndex] = None

    # ===== MCP 注入 =====
    def set_router(self, router):
        self.router = router

    # ===== 工具：展平/文本化（兼容 llm_outline 的结构化分块） =====
    def _flatten_parts(self, parts):
        out = []
        stack = list(parts) if isinstance(parts, (list, tuple)) else [parts]
        while stack:
            x = stack.pop(0)
            if x is None:
                continue
            if isinstance(x, (list, tuple)):
                stack[0:0] = list(x)
            else:
                out.append(x)
        return out

    def _as_text(self, part):
        if isinstance(part, str):
            return part
        if isinstance(part, dict):
            for key in ("text", "content", "chunk", "body"):
                v = part.get(key)
                if isinstance(v, str) and v.strip():
                    return v
            head = part.get("title") or part.get("heading") or ""
            body = part.get("summary") or part.get("bullets") or part.get("items") or ""
            if isinstance(body, (list, tuple)):
                body = "\n".join(str(i) for i in body)
            s = f"{head}\n{body}".strip()
            return s if s else str(part)
        return str(part)

    # ===== 本地读取 =====
    def _read_pdf(self, path: Path) -> str:
        try:
            import fitz  # 懒加载，避免 DLL 在非 PDF 场景触发崩溃
        except Exception:
            return ""
        text = []
        with fitz.open(path) as doc:
            for page in doc:
                text.append(page.get_text())
        return "\n".join(text)

    def _read_docx(self, path: Path) -> str:
        try:
            import docx  # type: ignore
        except Exception:
            return ""
        try:
            d = docx.Document(str(path))
            return "\n".join(p.text for p in d.paragraphs)
        except Exception:
            return ""

    def _scan_local_sig(self) -> Tuple[List[Tuple[str, int, int]], List[Path]]:
        files: List[Path] = []
        sig_items: List[Tuple[str, int, int]] = []
        for p in self.cfg.docs_dir.rglob("*"):
            suf = p.suffix.lower()
            if suf not in {".txt", ".md", ".pdf", ".docx"}:
                continue
            try:
                st = p.stat()
            except FileNotFoundError:
                continue
            files.append(p)
            sig_items.append((str(p), int(st.st_mtime), int(st.st_size)))
        return sig_items, files

    # ===== MCP 读取（切割仍本地）=====
    async def build_index_from_docs_via_mcp(self):
        if not self.router:
            return None, None, None
        res = await self.router.call("file.read_dir", {
            "dir": str(self.cfg.docs_dir),
            "patterns": ["*.pdf", "*.txt", "*.md", "*.docx"],
            "recursive": True,
            "limit": None,
            "normalize": True
        })
        chunks: List[str] = []
        metas: List[Dict[str, Any]] = []
        sig_items: List[Tuple[str, int, int]] = []

        for f in (res.get("files", []) or []):
            meta = f.get("meta", {}) or {}
            if meta.get("error"):
                continue
            source = meta.get("source") or meta.get("path") or "unknown"
            mtime = int(meta.get("mtime", 0)) if isinstance(meta.get("mtime", 0), (int, float)) else 0
            size_meta = int(meta.get("size", 0)) if isinstance(meta.get("size", 0), (int, float)) else 0
            content = f.get("content", "") or ""
            size_for_sig = size_meta if size_meta > 0 else len(content)
            sig_items.append((str(source), mtime, int(size_for_sig)))
            if not content.strip():
                continue

            parts = self._flatten_parts(self._chunker.split(content))
            for i, ch in enumerate(parts):
                txt = self._as_text(ch)
                if not txt.strip():
                    continue
                metas.append({"text": txt, "source": source, "chunk_id": i})
                chunks.append(txt)

        docs_sig = _hash_list(sig_items)
        return chunks, metas, docs_sig

    # ===== 本地读取（切块）=====
    def build_index_from_docs_local(self):
        chunks: List[str] = []
        metas: List[Dict[str, Any]] = []
        sig_items, files = self._scan_local_sig()
        for p in files:
            suf = p.suffix.lower()
            try:
                if suf in {".txt", ".md"}:
                    text = p.read_text("utf-8", errors="ignore")
                elif suf == ".pdf":
                    text = self._read_pdf(p)
                elif suf == ".docx":
                    text = self._read_docx(p)
                else:
                    continue
            except Exception:
                continue
            parts = self._flatten_parts(self._chunker.split(text))
            for i, part in enumerate(parts):
                txt = self._as_text(part)
                if not txt.strip():
                    continue
                metas.append({"text": txt, "source": str(p), "chunk_id": i})
                chunks.append(txt)
        docs_sig = _hash_list(sig_items)
        return chunks, metas, docs_sig

    # ===== Backend 选择/加载 =====
    def _new_backend(self, dim: int) -> VectorIndex:
        # 环境变量可强制：ANN_BACKEND=hnsw/flat（默认 hnsw，失败自动回退 flat）
        pref = os.getenv("ANN_BACKEND", "hnsw").lower()
        if pref == "hnsw":
            try:
                return HNSWIndex(dim=dim, space=self._HNSW_SPACE, M=self._HNSW_M,
                                 ef_construction=self._HNSW_EFC, ef_search=self._HNSW_EFS)
            except Exception as e:
                print(f"⚠️ HNSW 不可用，回退 Flat。原因：{e}")
        return FlatIndex(dim=dim)

    def _load_backend_if_exists(self) -> bool:
        meta = self.index.get("hnsw") or {}
        kind = (meta.get("kind") or "hnsw").lower()
        dim = meta.get("dim")
        if not isinstance(dim, int) or dim <= 0:
            return False
        self._dim = dim
        backend: VectorIndex = HNSWIndex(dim=dim) if kind == "hnsw" else FlatIndex(dim=dim)
        try:
            backend.load(self._idx_prefix)
        except Exception as e:
            print("⚠️ 加载已存在索引失败，将重建：", e)
            return False
        self._backend = backend
        return True

    def _save_meta_json(self, *, built_by: str, docs_sig: str, metas: List[Dict[str, Any]], dim: int, kind: str):
        self._dim = int(dim)
        self.index = {
            "metas": metas,
            "built_by": built_by,
            "docs_sig": docs_sig,
            "hnsw": {
                "kind": kind,  # 'hnsw' | 'flat'
                "space": self._HNSW_SPACE,
                "M": self._HNSW_M,
                "ef_construction": self._HNSW_EFC,
                "ef_search": self._HNSW_EFS,
                "count": len(metas),
                "dim": self._dim,
            }
        }
        self._json_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._json_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.index, ensure_ascii=False), "utf-8")
        tmp.replace(self._json_path)

    # ===== 外部文本重建 =====
    async def rebuild_from_texts(self, texts: List[str], metadatas: List[Dict[str, Any]],
                                 batch_size: int = 16, built_by: str = "local", docs_sig: str = ""):
        if not texts:
            # 清空
            self._save_meta_json(built_by=built_by, docs_sig=docs_sig, metas=[], dim=self._dim or 0, kind="hnsw")
            # 删除旧索引文件
            for suf in (".bin", ".npz"):
                p = self._idx_prefix.with_suffix(suf)
                if p.exists():
                    try: p.unlink()
                    except Exception: pass
            self._backend = None
            return

        metas_out: List[Dict[str, Any]] = []
        ids: List[int] = []
        all_vecs: List[np.ndarray] = []

        # 嵌入
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            vecs = await self.emb.embed(batch)
            all_vecs.append(np.asarray(vecs, dtype=np.float32))
            metas_out.extend(metadatas[i:i + batch_size])
            ids.extend(list(range(i, i + len(batch))))

        X = np.vstack(all_vecs)
        # 清洗 NaN/Inf
        if not np.isfinite(X).all():
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

        ids_np = np.asarray(ids, dtype=np.int64)
        d = int(X.shape[1])

        backend = self._new_backend(d)
        backend.add_batch(X, ids_np)
        backend.save(self._idx_prefix)
        self._backend = backend

        self._save_meta_json(built_by=built_by, docs_sig=docs_sig, metas=metas_out, dim=d, kind=backend.kind)

    # ===== ensure_index（/reindex 会调用）=====
    async def ensure_index(self, batch_size: int = 16, force: bool = False):
        metas_old = self.index.get("metas") or []
        built_by_old = self.index.get("built_by")
        old_sig = self.index.get("docs_sig") or ""
        prefer_mcp = self.router is not None

        json_ok = bool(metas_old)
        idx_exists = self._idx_prefix.with_suffix(".bin").exists() or self._idx_prefix.with_suffix(".npz").exists()
        need_rebuild = force or (not idx_exists) or (not json_ok)

        if not need_rebuild:
            try:
                if prefer_mcp:
                    _c, _m, new_sig = await self.build_index_from_docs_via_mcp()
                else:
                    _c, _m, new_sig = self.build_index_from_docs_local()
            except Exception:
                new_sig = None
            if new_sig and old_sig and new_sig != old_sig:
                need_rebuild = True
            if prefer_mcp and built_by_old and "reader:local" in built_by_old:
                need_rebuild = True

        if not need_rebuild:
            if self._backend is None:
                self._load_backend_if_exists()
            return

        # ① 优先 MCP
        chunks = metas = docs_sig = None
        built_by_now = f"reader:local; chunker:{CHUNK.name}"
        if prefer_mcp:
            try:
                chunks, metas, docs_sig = await self.build_index_from_docs_via_mcp()
                if chunks is not None:
                    built_by_now = f"reader:mcp; chunker:{CHUNK.name}"
            except Exception as e:
                print("⚠️ MCP 构建失败，回退本地：", e)

        # ② 回退本地
        if not chunks:
            chunks, metas, docs_sig = self.build_index_from_docs_local()
            built_by_now = f"reader:local; chunker:{CHUNK.name}"

        if not chunks:
            # 空目录：写空索引并清理文件
            self._save_meta_json(built_by=built_by_now, docs_sig=docs_sig or "", metas=[], dim=self._dim or 0, kind="hnsw")
            for suf in (".bin", ".npz"):
                p = self._idx_prefix.with_suffix(suf)
                if p.exists():
                    try: p.unlink()
                    except Exception: pass
            self._backend = None
            return

        # 复用旧 embeddings（如存在且长度匹配）
        legacy_embeddings = None
        if isinstance(self.index, dict) and isinstance(self.index.get("embeddings"), list):
            try:
                if len(self.index["embeddings"]) == len(metas):
                    legacy_embeddings = np.asarray(self.index["embeddings"], dtype=np.float32)
            except Exception:
                legacy_embeddings = None

        if legacy_embeddings is not None:
            X = legacy_embeddings
        else:
            all_vecs: List[np.ndarray] = []
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                vecs = await self.emb.embed(batch)
                all_vecs.append(np.asarray(vecs, dtype=np.float32))
            X = np.vstack(all_vecs)

        if not np.isfinite(X).all():
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

        ids_np = np.arange(len(X), dtype=np.int64)
        d = int(X.shape[1])

        backend = self._new_backend(d)
        backend.add_batch(X, ids_np)
        backend.save(self._idx_prefix)
        self._backend = backend

        self._save_meta_json(built_by=built_by_now, docs_sig=docs_sig or "", metas=metas, dim=d, kind=backend.kind)
        if "embeddings" in self.index:
            try:
                self.index.pop("embeddings", None)
                self._json_path.write_text(json.dumps(self.index, ensure_ascii=False), "utf-8")
            except Exception:
                pass

    # ===== 查询 =====
    async def topk(self, query: str, k: int):
        await self.ensure_index()
        metas = self.index.get("metas") or []
        if not metas:
            return []

        if self._backend is None and not self._load_backend_if_exists():
            return []

        qv = (await self.emb.embed([query]))[0]
        q = np.asarray(qv, dtype=np.float32).reshape(1, -1)
        if not np.isfinite(q).all():
            q = np.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0)

        labels, _ = self._backend.search(q, k=int(k))
        raw_ids = labels[0].tolist() if len(labels) else []

        # 保序去重
        seen, ordered = set(), []
        for i in raw_ids:
            if i not in seen:
                seen.add(i)
                ordered.append(i)

        out = []
        for i in ordered:
            if 0 <= i < len(metas):
                m = metas[i]
                if "text" not in m or not isinstance(m["text"], str) or not m["text"].strip():
                    fb = m.get("content") or m.get("chunk") or m.get("body") or ""
                    if not isinstance(fb, str):
                        fb = str(fb)
                    m = {**m, "text": fb}
                out.append(m)
                if len(out) >= k:
                    break
        return out
