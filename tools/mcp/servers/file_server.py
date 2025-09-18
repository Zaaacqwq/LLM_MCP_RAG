# tools/file_server.py
import sys, json
from pathlib import Path

try:
    import fitz  # PyMuPDF for PDF
except Exception:
    fitz = None

def _send(result=None, id=None, error=None):
    msg = {"jsonrpc": "2.0", "id": id}
    if error is not None:
        msg["error"] = {"message": str(error)}
    else:
        msg["result"] = result or {}
    print(json.dumps(msg, ensure_ascii=False), flush=True)

def _read_text(p: Path) -> str:
    return p.read_text("utf-8", errors="ignore")

def _read_pdf(p: Path) -> str:
    if not fitz:
        raise RuntimeError("PyMuPDF (fitz) not installed")
    text = []
    with fitz.open(str(p)) as doc:
        for page in doc:
            text.append(page.get_text())
    return "\n".join(text)

def _read_docx(p: Path) -> str:
    try:
        import docx
    except Exception as e:
        raise RuntimeError(f"python-docx not installed: {e}")
    d = docx.Document(str(p))
    return "\n".join([para.text for para in d.paragraphs])

def _file_read_dir(args: dict) -> dict:
    """
    仅用于读取文件，返回每个文件的 meta + content。
    不做切割；切割交给 agent.chunkers 中的 get_chunker。
    """
    base = Path(args.get("dir", "../.."))
    patterns = args.get("patterns", ["*.txt", "*.md", "*.pdf"])
    recursive = bool(args.get("recursive", True))
    limit = args.get("limit")
    normalize = bool(args.get("normalize", True))

    files = []
    for pat in patterns:
        glob_pat = f"**/{pat}" if recursive else pat
        files += [p for p in base.glob(glob_pat) if p.is_file()]
    if isinstance(limit, int):
        files = files[:limit]

    out = {"files": []}
    for p in files:
        meta = {"source": str(p), "path": str(p)}
        try:
            st = p.stat()
            meta.update({"mtime": int(st.st_mtime), "size": int(st.st_size)})
            sfx = p.suffix.lower()
            if sfx in (".txt", ".md"):
                content = _read_text(p)
            elif sfx == ".pdf":
                content = _read_pdf(p)
            elif sfx == ".docx":
                try:
                    content = _read_docx(p)
                except Exception as e:
                    meta["error"] = f"docx error: {e}"
                    content = ""
            else:
                meta["error"] = "unsupported"
                content = ""
            if normalize and content:
                # 简单的换行归一/去 BOM 等
                content = content.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff")
        except Exception as e:
            meta["error"] = str(e)
            content = ""
        out["files"].append({"meta": meta, "content": content})
    return out

# ---- MCP handlers ----

def _handle_initialize(params, id):
    _send({"server": "file-server", "version": "0.2", "capability": "read-only"}, id)

def _handle_tools_list(params, id):
    # 只暴露读取工具；不再提供 file.chunk
    _send({"tools": ["file.read_dir"]}, id)

def _handle_tools_call(params, id):
    name = params.get("name", "")
    args = params.get("arguments") or {}
    try:
        if name == "file.read_dir":
            res = _file_read_dir(args)
        else:
            raise RuntimeError("unknown tool")
        _send(res, id)
    except Exception as e:
        _send(None, id, error=str(e))

def main():
    handlers = {
        "initialize": _handle_initialize,
        "tools/list": _handle_tools_list,
        "tools/call": _handle_tools_call,
    }
    for line in sys.stdin:
        raw = line.strip()
        if not raw:
            continue
        try:
            msg = json.loads(raw)
        except Exception:
            print(f"[file-server] non-JSON: {raw}", file=sys.stderr, flush=True)
            continue
        mid = msg.get("id")
        method = msg.get("method")
        params = msg.get("params") or {}
        h = handlers.get(method)
        if not h:
            _send(None, mid, error=f"unknown method {method}")
            continue
        h(params, mid)

if __name__ == "__main__":
    main()
