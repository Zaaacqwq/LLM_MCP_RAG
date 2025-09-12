# agent/mcp_client.py
import asyncio
import json
import sys
import uuid
import subprocess
from typing import Any, Dict, Optional


class JSONRPCError(Exception):
    pass


class MCPClient:
    def __init__(
        self,
        command: list[str],
        *,
        start_timeout: float = 30.0,
        init_timeout: float = 30.0,
        call_timeout: float = 45.0,
        print_logs: bool = True,
    ):
        self.command = command
        self.proc: Optional[asyncio.subprocess.Process] = None
        self.pending: Dict[str, asyncio.Future] = {}
        self.reader_task: Optional[asyncio.Task] = None

        self.start_timeout = start_timeout
        self.init_timeout = init_timeout
        self.call_timeout = call_timeout
        self.print_logs = print_logs

    async def start(self, timeout: Optional[float] = None):
        """启动 MCP 子进程并拉起 reader。"""
        if not self.command:
            raise JSONRPCError("Empty MCP command")

        # Windows: 隐藏子进程窗口（可选）
        creationflags = 0
        if sys.platform.startswith("win"):
            creationflags |= subprocess.CREATE_NO_WINDOW

        async def _spawn():
            self.proc = await asyncio.create_subprocess_exec(
                *self.command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=sys.stderr,  # 直通当前控制台，便于定位
                creationflags=creationflags if sys.platform.startswith("win") else 0,
            )
            # 启动 reader
            self.reader_task = asyncio.create_task(self._reader())

        to = timeout or self.start_timeout
        try:
            await asyncio.wait_for(_spawn(), timeout=to)
        except asyncio.TimeoutError as e:
            raise TimeoutError(f"MCP start timeout after {to}s: {self.command}") from e

    async def _reader(self):
        """持续读取子进程 stdout；按行解析 JSON-RPC；非 JSON 行当作日志打印。"""
        assert self.proc and self.proc.stdout
        while True:
            try:
                line = await self.proc.stdout.readline()
                if not line:
                    # 子进程退出；把所有 pending 置异常
                    err = JSONRPCError("MCP process exited / pipe closed")
                    for mid, fut in list(self.pending.items()):
                        if not fut.done():
                            fut.set_exception(err)
                    self.pending.clear()
                    break

                raw = line.decode("utf-8", errors="ignore").strip()
                if not raw:
                    continue

                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    if self.print_logs:
                        print(f"[MCP][log] {raw}")
                    continue

                # JSON-RPC 响应
                if "id" in msg:
                    mid = msg["id"]
                    fut = self.pending.pop(mid, None)
                    if fut and not fut.done():
                        fut.set_result(msg)
                    continue

                # JSON-RPC 通知/事件（可按需扩展）
                if self.print_logs:
                    method = msg.get("method", "<notify>")
                    print(f"[MCP][notify] {method}: {msg.get('params')}")

            except Exception as e:
                # 防御式：reader 内部异常不应静默
                if self.print_logs:
                    print(f"[MCP][reader-error] {e!r}")
                err = JSONRPCError(f"MCP reader crashed: {e!r}")
                for mid, fut in list(self.pending.items()):
                    if not fut.done():
                        fut.set_exception(err)
                self.pending.clear()
                break

    async def _call(self, method: str, params: Any | None = None, *, timeout: Optional[float] = None):
        """发送 JSON-RPC 请求并等待响应。"""
        if not self.proc or not self.proc.stdin:
            raise JSONRPCError("MCP process not started")

        mid = str(uuid.uuid4())
        req = {"jsonrpc": "2.0", "id": mid, "method": method, "params": params or {}}

        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        self.pending[mid] = fut

        try:
            payload = (json.dumps(req, ensure_ascii=False) + "\n").encode("utf-8")
            self.proc.stdin.write(payload)
            await self.proc.stdin.drain()
        except Exception as e:
            # 写入失败，清理 pending
            self.pending.pop(mid, None)
            raise JSONRPCError(f"Failed to write to MCP stdin: {e!r}")

        # 等待响应
        to = timeout or self.call_timeout
        try:
            resp = await asyncio.wait_for(fut, timeout=to)
        except asyncio.TimeoutError as e:
            # 超时清理，避免 future 泄漏
            self.pending.pop(mid, None)
            raise TimeoutError(f"MCP call timeout after {to}s: {method}") from e

        # 错误响应
        if "error" in resp and resp["error"] is not None:
            err_obj = resp["error"]
            # 容忍 error 是对象/字符串两种形态
            msg = err_obj if isinstance(err_obj, str) else err_obj.get("message", str(err_obj))
            raise JSONRPCError(f"{method} error: {msg}")

        return resp.get("result")

    async def initialize(self, client_info: dict, *, timeout: Optional[float] = None):
        return await self._call("initialize", client_info, timeout=timeout or self.init_timeout)

    async def tools_list(self):
        return await self._call("tools/list", {})

    async def tools_call(self, name: str, arguments: dict):
        return await self._call("tools/call", {"name": name, "arguments": arguments})

    async def stop(self):
        """停止 reader 并终止子进程。"""
        # 先取消 reader
        if self.reader_task and not self.reader_task.done():
            self.reader_task.cancel()
            try:
                await self.reader_task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                if self.print_logs:
                    print(f"[MCP] reader stop error: {e!r}")

        # 再杀子进程
        if self.proc:
            try:
                self.proc.kill()
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(self.proc.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                if self.print_logs:
                    print("[MCP] process did not exit in time")
        self.proc = None
        self.pending.clear()
