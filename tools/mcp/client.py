# tools/mcp/client.py
import os, sys, json, subprocess, threading, uuid
from typing import Dict, Any, Optional

class MCPUnavailable(Exception): ...
class MCPError(Exception): ...

class _MCPProc:
    """
    极简按行 JSON-RPC 客户端（单进程、串行调用即可）。
    """
    def __init__(self, argv: list[str]):
        self.argv = argv
        self.p: Optional[subprocess.Popen] = None
        self.lock = threading.Lock()
        self._next_id = 0

    def start(self):
        if self.p is not None:
            return
        self.p = subprocess.Popen(
            self.argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,  # 如需看日志可改成 sys.stderr
            text=True,
            encoding="utf-8",
        )
        # 基本握手
        self._rpc("initialize", {})

    def _rpc(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if self.p is None or self.p.poll() is not None:
            raise MCPUnavailable(f"MCP server not running: {' '.join(self.argv)}")
        with self.lock:
            self._next_id += 1
            mid = self._next_id
            payload = {"jsonrpc": "2.0", "id": mid, "method": method, "params": params}
            assert self.p.stdin is not None and self.p.stdout is not None
            self.p.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self.p.stdin.flush()
            # 逐行读取，直到拿到对应 id（此实现假设串行调用）
            line = self.p.stdout.readline()
            if not line:
                raise MCPUnavailable("MCP server closed pipe")
            msg = json.loads(line)
            if "error" in msg and msg["error"]:
                raise MCPError(str(msg["error"].get("message", msg["error"])))
            return msg.get("result") or {}

    def tools_list(self):
        res = self._rpc("tools/list", {})
        # 兼容两种返回格式
        tools = res.get("tools", res)
        names = []
        for t in tools:
            if isinstance(t, str):
                names.append(t)
            elif isinstance(t, dict) and "name" in t:
                names.append(t["name"])
        return names

    def call(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        return self._rpc("tools/call", {"name": name, "arguments": arguments})

    def close(self):
        try:
            if self.p and self.p.poll() is None:
                self.p.terminate()
        except Exception:
            pass


class MCPClient:
    """
    兼容你原有占位实现的 API：
      - has_tool(name) -> bool
      - call(name, args) -> dict
    使用方法：
      1) mcp = MCPClient(); mcp.start()
      2) mcp.call("math.solve_equation", {"expr":"x^2+3x+2"})
    """
    def __init__(self):
        self._tools = set()
        self._ready = False

        base = os.path.dirname(os.path.abspath(__file__))
        servers = os.path.join(base, "servers")
        py = sys.executable

        self._file = _MCPProc([py, os.path.join(servers, "file_server.py")])   # file.read_dir :contentReference[oaicite:3]{index=3}
        self._math = _MCPProc([py, os.path.join(servers, "math_server.py")])   # math.*       :contentReference[oaicite:4]{index=4}
        self._code = _MCPProc([py, os.path.join(servers, "code_server.py")])   # *.run_code   :contentReference[oaicite:5]{index=5}

    def start(self):
        """
        显式启动三台 MCP 进程并收集可用工具。
        """
        try:
            self._file.start()
            self._math.start()
            self._code.start()
            # 汇总工具清单
            for t in self._file.tools_list():
                self._tools.add(t)
            for t in self._math.tools_list():
                # 工具清单可能返回 dict，需要提取 name；_MCPProc 已兼容
                self._tools.add(t)
            for t in self._code.tools_list():
                self._tools.add(t)
            self._ready = True
        except Exception as e:
            # 启动失败保持占位行为
            self._ready = False
            raise MCPUnavailable(f"Failed to start MCP servers: {e}")

    def has_tool(self, name: str) -> bool:
        return name in self._tools

    def _ensure_ready(self):
        if not self._ready:
            raise MCPUnavailable("MCP not configured or not started. Call MCPClient.start() first.")

    def call(self, name: str, args: dict) -> dict:
        """
        路由规则：
          file.*  -> file_server
          math.*  -> math_server
          *.run_code -> code_server
        """
        self._ensure_ready()
        if name.startswith("file."):
            return self._file.call(name, args)
        if name.startswith("math."):
            return self._math.call(name, args)
        if name.endswith(".run_code"):
            return self._code.call(name, args)
        raise MCPUnavailable(f"Unknown MCP tool: {name}")

    def close(self):
        for proc in (self._file, self._math, self._code):
            proc.close()
