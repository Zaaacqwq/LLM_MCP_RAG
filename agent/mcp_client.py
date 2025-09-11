import asyncio, json, uuid, sys
from typing import Any

class JSONRPCError(Exception): pass

class MCPClient:
    def __init__(self, command: list[str]):
        self.command = command
        self.proc = None
        self.pending = {}

    async def start(self):
        if not self.command:
            return
        self.proc = await asyncio.create_subprocess_exec(
            *self.command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=sys.stderr
        )
        asyncio.create_task(self._reader())

    async def _reader(self):
        # 简单行流：一行一个 JSON（你的 server 也按行 flush）
        while True:
            line = await self.proc.stdout.readline()
            if not line:
                break
            msg = json.loads(line.decode("utf-8"))
            if "id" in msg and (mid := msg["id"]) in self.pending:
                fut = self.pending.pop(mid)
                fut.set_result(msg)
            # 如果要处理通知/事件，可在此扩展

    async def _call(self, method: str, params: Any | None=None):
        if self.proc is None:
            raise JSONRPCError("MCP process not started")
        mid = str(uuid.uuid4())
        req = {"jsonrpc": "2.0", "id": mid, "method": method, "params": params or {}}
        fut = asyncio.get_event_loop().create_future()
        self.pending[mid] = fut
        self.proc.stdin.write((json.dumps(req) + "\n").encode("utf-8"))
        await self.proc.stdin.drain()
        resp = await asyncio.wait_for(fut, timeout=30)
        if "error" in resp:
            raise JSONRPCError(resp["error"])
        return resp.get("result")

    async def initialize(self, client_info: dict):
        return await self._call("initialize", client_info)

    async def tools_list(self):
        return await self._call("tools/list")

    async def tools_call(self, name: str, arguments: dict):
        return await self._call("tools/call", {"name": name, "arguments": arguments})

    async def stop(self):
        if self.proc:
            self.proc.kill()
            await self.proc.wait()
