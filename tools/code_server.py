# tools/code_server.py
import sys, os, json, tempfile, shutil, time, textwrap, subprocess, uuid
from typing import Dict, Any, List, Optional

# =============== 配置 ===============
DEFAULT_TIMEOUT_SEC = 3.0          # 运行/编译超时（秒）
MAX_OUTPUT_BYTES   = 200_000       # 截断输出上限（单流）
PYTHON_EXE         = sys.executable # 用当前解释器跑 python

# Windows 兼容：隐藏可能弹窗
CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0

# =============== JSON-RPC公共 ===============
def reply(msg_id, result=None, error=None):
    resp = {"jsonrpc": "2.0", "id": msg_id}
    if error is not None:
        resp["error"] = {"code": -32000, "message": str(error)}
    else:
        resp["result"] = result
    print(json.dumps(resp, ensure_ascii=False), flush=True)

# =============== 工具基础 ===============
def _truncate(b: bytes, limit: int) -> str:
    if b is None:
        return ""
    if len(b) <= limit:
        return b.decode(errors="replace")
    return (b[:limit].decode(errors="replace")) + f"\n\n[TRUNCATED: {len(b)-limit} bytes omitted]"

def _run(cmd: List[str], cwd: str, timeout: float) -> Dict[str, Any]:
    t0 = time.time()
    try:
        p = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            env={**os.environ, "PYTHONWARNINGS": "ignore"},
            creationflags=CREATE_NO_WINDOW
        )
        try:
            out, err = p.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            # 杀整个进程树（Windows 简化处理：terminate -> kill）
            try:
                p.terminate()
            except Exception:
                pass
            try:
                p.kill()
            except Exception:
                pass
            raise TimeoutError(f"Process timed out after {timeout:.1f}s")
        rc = p.returncode
        elapsed = int((time.time() - t0) * 1000)
        return {
            "stdout": _truncate(out or b"", MAX_OUTPUT_BYTES),
            "stderr": _truncate(err or b"", MAX_OUTPUT_BYTES),
            "exit_code": rc,
            "time_ms": elapsed
        }
    except TimeoutError as e:
        elapsed = int((time.time() - t0) * 1000)
        return {"stdout": "", "stderr": str(e), "exit_code": -1, "time_ms": elapsed}
    except FileNotFoundError as e:
        return {"stdout": "", "stderr": f"Executable not found: {e}", "exit_code": -2, "time_ms": int((time.time()-t0)*1000)}
    except Exception as e:
        return {"stdout": "", "stderr": f"Subprocess error: {e}", "exit_code": -3, "time_ms": int((time.time()-t0)*1000)}

def _ensure_tmpdir() -> str:
    d = os.path.join(tempfile.gettempdir(), f"mcp-code-{uuid.uuid4().hex[:8]}")
    os.makedirs(d, exist_ok=True)
    return d

def _list_files(dirpath: str) -> List[str]:
    paths = []
    for root, _, files in os.walk(dirpath):
        for f in files:
            ap = os.path.join(root, f)
            rel = os.path.relpath(ap, dirpath)
            paths.append(rel.replace("\\", "/"))
    return sorted(paths)

# =============== 编译器探测 ===============
def _which(names: List[str]) -> Optional[str]:
    for n in names:
        p = shutil.which(n)
        if p:
            return p
    return None

# =============== 语言实现 ===============
def tool_python_run(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    args: {"code": "<python code string>", "timeout": float?}
    """
    code = args.get("code", "")
    timeout = float(args.get("timeout", DEFAULT_TIMEOUT_SEC))
    if not code.strip():
        raise ValueError("code 不能为空。")

    # ⚠️ 安全提示：这是本地执行器，只用于受信代码！
    # 如需更强隔离请考虑：容器/命名空间/禁网/沙箱等。
    work = _ensure_tmpdir()
    main_py = os.path.join(work, "main.py")
    with open(main_py, "w", encoding="utf-8") as f:
        f.write(code)

    run_res = _run([PYTHON_EXE, "main.py"], cwd=work, timeout=timeout)
    run_res["workdir"] = work.replace("\\", "/")
    run_res["files"] = _list_files(work)
    return run_res

def tool_c_run(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    args: {"code": "<c code>", "timeout": float?}
    依次尝试编译器：gcc, clang
    """
    code = args.get("code", "")
    timeout = float(args.get("timeout", DEFAULT_TIMEOUT_SEC))
    if not code.strip():
        raise ValueError("code 不能为空。")

    cc = _which(["gcc", "clang"])
    if not cc:
        raise RuntimeError("未找到 C 编译器：请安装 gcc 或 clang 并加入 PATH。")

    work = _ensure_tmpdir()
    src = os.path.join(work, "main.c")
    exe = os.path.join(work, "a.exe" if os.name == "nt" else "a.out")
    with open(src, "w", encoding="utf-8") as f:
        f.write(code)

    # 编译
    compile_res = _run([cc, "main.c", "-O2", "-pipe", "-static-libgcc", "-o", exe], cwd=work, timeout=timeout)
    # 某些平台不支持 -static-libgcc；非致命
    if compile_res["exit_code"] != 0:
        # 再试一次不带静态链接
        compile_res = _run([cc, "main.c", "-O2", "-pipe", "-o", exe], cwd=work, timeout=timeout)

    result = {"compile": compile_res, "workdir": work.replace("\\", "/")}
    result["files"] = _list_files(work)
    if compile_res["exit_code"] == 0:
        run_res = _run([exe], cwd=work, timeout=timeout)
        result["run"] = run_res
    return result

def tool_cpp_run(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    args: {"code": "<cpp code>", "timeout": float?}
    依次尝试编译器：g++, clang++
    """
    code = args.get("code", "")
    timeout = float(args.get("timeout", DEFAULT_TIMEOUT_SEC))
    if not code.strip():
        raise ValueError("code 不能为空。")

    cxx = _which(["g++", "clang++"])
    if not cxx:
        raise RuntimeError("未找到 C++ 编译器：请安装 g++ 或 clang++ 并加入 PATH。")

    work = _ensure_tmpdir()
    src = os.path.join(work, "main.cpp")
    exe = os.path.join(work, "a.exe" if os.name == "nt" else "a.out")
    with open(src, "w", encoding="utf-8") as f:
        f.write(code)

    compile_res = _run([cxx, "main.cpp", "-std=c++17", "-O2", "-pipe", "-o", exe], cwd=work, timeout=timeout)
    result = {"compile": compile_res, "workdir": work.replace("\\", "/")}
    result["files"] = _list_files(work)
    if compile_res["exit_code"] == 0:
        run_res = _run([exe], cwd=work, timeout=timeout)
        result["run"] = run_res
    return result

def tool_java_run(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    args: {"code": "<java code>", "timeout": float?}
    需要：javac, java 在 PATH 中
    约定主类名：Main（public class Main { public static void main... }）
    """
    code = args.get("code", "")
    timeout = float(args.get("timeout", DEFAULT_TIMEOUT_SEC))
    if not code.strip():
        raise ValueError("code 不能为空。")

    javac = _which(["javac"])
    java = _which(["java"])
    if not (javac and java):
        raise RuntimeError("未找到 Java 工具链：需要安装并将 javac/java 加入 PATH。")

    work = _ensure_tmpdir()
    src = os.path.join(work, "Main.java")
    with open(src, "w", encoding="utf-8") as f:
        f.write(code)

    compile_res = _run([javac, "Main.java"], cwd=work, timeout=timeout)
    result = {"compile": compile_res, "workdir": work.replace("\\", "/")}
    result["files"] = _list_files(work)
    if compile_res["exit_code"] == 0:
        run_res = _run([java, "Main"], cwd=work, timeout=timeout)
        result["run"] = run_res
    return result

# =============== JSON-RPC分发 ===============
def handle_call(name: str, args: dict):
    if name == "python.run_code":
        return tool_python_run(args)
    elif name == "c.run_code":
        return tool_c_run(args)
    elif name == "cpp.run_code":
        return tool_cpp_run(args)
    elif name == "java.run_code":
        return tool_java_run(args)
    else:
        raise ValueError(f"未知工具: {name}")

def main():
    for line in sys.stdin:
        if not line.strip():
            continue
        req = json.loads(line)
        mid = req.get("id")
        method = req.get("method")
        params = req.get("params", {}) or {}
        try:
            if method == "initialize":
                reply(mid, {"server": "code-tools", "version": "0.1"})
            elif method == "tools/list":
                reply(mid, {"tools": [
                    {"name": "python.run_code", "desc": "运行 Python 代码"},
                    {"name": "c.run_code", "desc": "编译并运行 C 代码"},
                    {"name": "cpp.run_code", "desc": "编译并运行 C++ 代码"},
                    {"name": "java.run_code", "desc": "编译并运行 Java 代码（主类 Main）"},
                ]})
            elif method == "tools/call":
                name = params.get("name")
                args = params.get("arguments", {}) or {}
                result = handle_call(name, args)
                reply(mid, result)
            else:
                reply(mid, error=f"未知方法: {method}")
        except Exception as e:
            reply(mid, error=str(e))

if __name__ == "__main__":
    main()
