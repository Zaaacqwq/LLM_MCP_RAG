# tools/tool_router.py
from tools.mcp.client import MCPClient
mcp = MCPClient(); mcp.start()

def call(tool_name: str, args: dict):
    if tool_name.startswith("math.") or tool_name.endswith(".run_code") or tool_name.startswith("file."):
        return mcp.call(tool_name, args)
    raise ValueError(f"Unknown tool: {tool_name}")
