import re

class ToolRouter:
    def __init__(self, mcp_client):
        self.mcp = mcp_client
        self.tools = []

    async def refresh(self):
        if not self.mcp:
            return
        res = await self.mcp.tools_list()
        self.tools = res.get("tools", [])

    def need_tool(self, user_text: str) -> bool:
        if not self.mcp:           # <<< 关键：没有 MCP，就不走工具
            return False
        return bool(re.search(r"\b(calc|run:|search)\b", user_text, re.I))

    def pick(self, user_text: str):
        if "calc" in user_text.lower():
            return "calculator.add", {"a": 3, "b": 5}
        if user_text.lower().startswith("run:"):
            cmd = user_text.split(":",1)[1].strip()
            return "shell.run", {"cmd": cmd}
        return None, None
