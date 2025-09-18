# retriever/reader.py
from typing import List, Dict
from tools.mcp.client import MCPClient

def read_dir_via_mcp(docs_dir: str, patterns: list[str] | None = None, recursive: bool = True) -> List[Dict[str, str]]:
    patterns = patterns or ["**/*.txt", "**/*.md", "**/*.pdf", "**/*.docx"]
    mcp = MCPClient(); mcp.start()
    res = mcp.call("file.read_dir", {"dir": docs_dir, "patterns": patterns, "recursive": recursive})
    out = []
    for it in res.get("files", []):
        # 兼容两种返回形态：
        # 1) 旧：{"path": "...", "text": "...", "error"?}
        # 2) 新：{"meta": {"path": "...", "source": "...", "error"?}, "content": "..."}
        meta = it.get("meta", it)
        text = it.get("content", it.get("text", ""))
        path = meta.get("path") or meta.get("source") or it.get("path", "")
        err  = meta.get("error") or it.get("error")
        if err or not text:
            continue
        out.append({"path": path, "text": text})
    return out
