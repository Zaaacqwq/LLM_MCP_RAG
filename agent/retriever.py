# agent/retriever.py
from __future__ import annotations
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import fitz  # PyMuPDF

from agent.embeddings import Embeddings, cosine_sim
from agent.config import RAGConfig


def simple_split(text: str, size: int, overlap: int):
    size = max(1, int(size))
    overlap = max(0, int(overlap))
    if overlap >= size:
        overlap = size - 1
    chunks, n, start, MAX = [], len(text), 0, 10000
    while start < n and len(chunks) < MAX:
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


def _hash_list(items: List[Tuple[str, int, int]]) -> str:
    """对 (key, mtime, size) 列表做稳定哈希，作为 docs 签名。"""
    h = hashlib.sha256()
    for k, m, s in sorted(items, key=lambda x: x[0]):
        h.update(str(k).encode("utf-8")); h.update(b"|")
        h.update(str(int(m)).encode("utf-8")); h.update(b"|")
        h.update(str(int(s)).encode("utf-8")); h.update(b"\n")
    return h.hexdigest()


class Retriever:
    """
    读取 → 分块 → 向量化 → 索引落盘（data/index.json）
    - 若注入了 self.router（JSON-RPC MCP 路由）：优先用 file_server 工具
    - 否则回退到本地实现（txt/md 直接读，pdf 用 fitz）
    额外能力：
      - built_by: "mcp" / "local"
      - docs_sig: 基于 (path/mtime/size) 的文档签名
      - force: 强制重建
      - 自动发现文档变化并重建
    """
    def __init__(self, cfg: RAGConfig, emb: Embeddings):
        self.cfg = cfg
        self.emb = emb
        self.index: Dict[str, Any] = {"embeddings": [], "metas": []}
        self.router = None  # 由 CLI 初始化 MCP 后注入

        # 尝试加载已有索引
        try:
            if cfg.index_path.exists() and cfg.index_path.stat().st_size > 0:
                self.index = json.loads(cfg.index_path.read_text("utf-8"))
        except Exception as e:
            print("⚠️ index.json 无效，将重建:", e)
            self.index = {"embeddings": [], "metas": []}

    def set_router(self, router):
        """CLI 初始化 MCP 后调用：retr.set_router(router)"""
        self.router = router

    # ---------- 本地读取 ----------
    def _read_pdf(self, path: Path) -> str:
        text = []
        with fitz.open(path) as doc:
            for page in doc:
                text.append(page.get_text())
        return "\n".join(text)

    def _scan_local_sig(self) -> Tuple[List[Tuple[str, int, int]], List[Path]]:
        files: List[Path] = []
        sig_items: List[Tuple[str, int, int]] = []
        for p in self.cfg.docs_dir.rglob("*"):
            suf = p.suffix.lower()
            if suf not in {".txt", ".md", ".pdf"}:
                continue
            try:
                st = p.stat()
            except FileNotFoundError:
                continue
            files.append(p)
            sig_items.append((str(p), int(st.st_mtime), int(st.st_size)))
        return sig_items, files

    # ---------- MCP 路径 ----------
    async def build_index_from_docs_via_mcp(self) -> tuple[
        Optional[List[str]], Optional[List[Dict[str, Any]]], Optional[str]
    ]:
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

            ck = await self.router.call("file.chunk", {
                "text": content,
                "chunk_size": self.cfg.chunk_size,
                "chunk_overlap": self.cfg.chunk_overlap,
                "preserve_newlines": True
            })
            for i, ch in enumerate(ck.get("chunks", [])):
                metas.append({"text": ch, "source": source, "chunk_id": i})
                chunks.append(ch)

        docs_sig = _hash_list(sig_items)
        return chunks, metas, docs_sig

    # ---------- 本地路径 ----------
    def build_index_from_docs_local(self) -> tuple[List[str], List[Dict[str, Any]], str]:
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
                else:
                    continue
            except Exception:
                continue
            parts = simple_split(text, self.cfg.chunk_size, self.cfg.chunk_overlap)
            for i, part in enumerate(parts):
                metas.append({"text": part, "source": str(p), "chunk_id": i})
                chunks.append(part)
        docs_sig = _hash_list(sig_items)
        return chunks, metas, docs_sig

    # ---------- 索引确保 ----------
    async def ensure_index(self, batch_size: int = 16, force: bool = False):
        """
        触发重建条件：
          - force=True
          - 未构建
          - 文档签名变化
          - 现在有 MCP，但旧索引 built_by 为 local
        """
        built = bool(self.index.get("embeddings"))
        built_by_old = self.index.get("built_by")
        old_sig = self.index.get("docs_sig")
        prefer_mcp = self.router is not None

        if built and not force:
            need_rebuild = False
            if prefer_mcp and built_by_old == "local":
                need_rebuild = True
            try:
                if prefer_mcp:
                    _c, _m, new_sig = await self.build_index_from_docs_via_mcp()
                else:
                    _c, _m, new_sig = self.build_index_from_docs_local()
            except Exception:
                new_sig = None
            if old_sig and new_sig and old_sig != new_sig:
                need_rebuild = True
            if not need_rebuild:
                return  # 无需重建

        # ① 尝试 MCP
        chunks = metas = docs_sig = None
        built_by_now = "local"
        if prefer_mcp:
            try:
                chunks, metas, docs_sig = await self.build_index_from_docs_via_mcp()
                if chunks:
                    built_by_now = "mcp"
            except Exception as e:
                print("⚠️ MCP 构建索引失败，自动回退本地读取：", e)

        # ② 回退本地
        if not chunks:
            chunks, metas, docs_sig = self.build_index_from_docs_local()
            built_by_now = "local"

        # 空索引也要落盘（带上来源和签名）
        if not chunks:
            self.index = {"embeddings": [], "metas": [], "built_by": built_by_now, "docs_sig": docs_sig or ""}
            self.cfg.index_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.cfg.index_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(self.index, ensure_ascii=False), "utf-8")
            tmp.replace(self.cfg.index_path)
            return

        embeddings: List[List[float]] = []
        metas_out: List[Dict[str, Any]] = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            vecs = await self.emb.embed(batch)
            embeddings.extend(vecs)
            metas_out.extend(metas[i:i + batch_size])
            # 分批落盘
            self.index = {
                "embeddings": embeddings,
                "metas": metas_out,
                "built_by": built_by_now,
                "docs_sig": docs_sig or ""
            }
            self.cfg.index_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.cfg.index_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(self.index, ensure_ascii=False), "utf-8")
            tmp.replace(self.cfg.index_path)

    # ---------- 查询 ----------
    async def topk(self, query: str, k: int):
        await self.ensure_index()
        if not self.index.get("embeddings"):
            return []
        qv = (await self.emb.embed([query]))[0]
        scored = [(cosine_sim(qv, vec), meta) for vec, meta in zip(self.index["embeddings"], self.index["metas"])]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [meta for _, meta in scored[:k]]
