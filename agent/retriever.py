# agent/retriever.py
from __future__ import annotations
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import fitz  # PyMuPDF

from agent.embeddings import Embeddings, cosine_sim
from agent.config import RAGConfig, CHUNK, LLMConfig
from agent.chunkers import get_chunker
from agent.llm import LLM


def _hash_list(items: List[Tuple[str, int, int]]) -> str:
    """对 (key, mtime, size) 列表做稳定哈希，作为 docs 签名。"""
    import hashlib
    h = hashlib.sha256()
    for k, m, s in sorted(items, key=lambda x: x[0]):
        h.update(str(k).encode("utf-8")); h.update(b"|")
        h.update(str(int(m)).encode("utf-8")); h.update(b"|")
        h.update(str(int(s)).encode("utf-8")); h.update(b"\n")
    return h.hexdigest()


class Retriever:
    """
    读取 → 分块(本地 chunker/可选 LLM 大纲) → 向量化 → 索引落盘（data/index.json）
    - 若注入了 self.router（JSON-RPC MCP 路由）：仅用于 file.read_dir 读取原文
    - 切割统一走本地 chunker（不再调用 MCP 的 file.chunk）
    元数据：
      - built_by: "mcp"（表示读取来源于 MCP）或 "local"（本地读取）；与切割方式无关
      - docs_sig: 基于 (path/mtime/size) 的文档签名
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

        # 预创建一个 chunker（llm_outline 需要 LLM）
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

    def _read_docx(self, path: Path) -> str:
        try:
            import docx
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

    # ---------- MCP 读取路径（仅 read_dir；切割走本地 chunker） ----------
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

            # —— 使用本地 chunker 切割（替代 MCP file.chunk）——
            parts = self._chunker.split(content)
            for i, ch in enumerate(parts):
                metas.append({"text": ch, "source": source, "chunk_id": i})
                chunks.append(ch)

        docs_sig = _hash_list(sig_items)
        return chunks, metas, docs_sig

    # ---------- 本地读取路径（读本地文件 + 本地 chunker） ----------
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
                elif suf == ".docx":
                    text = self._read_docx(p)
                else:
                    continue
            except Exception:
                continue
            parts = self._chunker.split(text)
            for i, part in enumerate(parts):
                metas.append({"text": part, "source": str(p), "chunk_id": i})
                chunks.append(part)
        docs_sig = _hash_list(sig_items)
        return chunks, metas, docs_sig

    # ---------- 直接重建索引（供 orchestrator 调用） ----------
    async def rebuild_from_texts(
            self,
            texts: List[str],
            metadatas: List[Dict[str, Any]],
            batch_size: int = 16,
            built_by: str = "local",
            docs_sig: str = ""
    ):
        """用外部提供的切块重建索引（不走 MCP）。支持传入 built_by 与 docs_sig。"""
        if not texts:
            self.index = {
                "embeddings": [],
                "metas": [],
                "built_by": built_by,
                "docs_sig": docs_sig
            }
        else:
            embeddings: List[List[float]] = []
            metas_out: List[Dict[str, Any]] = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                vecs = await self.emb.embed(batch)
                embeddings.extend(vecs)
                metas_out.extend(metadatas[i:i + batch_size])
            self.index = {
                "embeddings": embeddings,
                "metas": metas_out,
                "built_by": built_by,
                "docs_sig": docs_sig or self.index.get("docs_sig", "")
            }

        self.cfg.index_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.cfg.index_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.index, ensure_ascii=False), "utf-8")
        tmp.replace(self.cfg.index_path)

    # ---------- 索引确保（保持与原语义一致，但切割全走本地） ----------
    async def ensure_index(self, batch_size: int = 16, force: bool = False):
        """
        触发重建条件：
          - force=True
          - 未构建
          - 文档签名变化
          - 现在有 MCP，但旧索引 built_by 为 local（优先用 MCP 读取）
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

        # ① 优先用 MCP 读取（切割仍是本地）
        chunks = metas = docs_sig = None
        built_by_now = "local"
        if prefer_mcp:
            try:
                chunks, metas, docs_sig = await self.build_index_from_docs_via_mcp()
                if chunks is not None:
                    built_by_now = "mcp"  # 表示“读取来自 MCP”
            except Exception as e:
                print("⚠️ MCP 构建索引失败，自动回退本地读取：", e)

        # ② 回退本地读取
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
