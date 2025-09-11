import os
import sys

from dotenv import load_dotenv
from dataclasses import dataclass, field
from pathlib import Path

load_dotenv()

# 读取 OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY. Please set it in .env or environment variables.")

@dataclass
class LLMConfig:
    provider: str = "openai"
    model: str = "gpt-4.1"
    api_base: str = "https://api.openai.com/v1"
    api_key: str = OPENAI_API_KEY

@dataclass
class RAGConfig:
    chunk_size: int = 600
    chunk_overlap: int = 120
    top_k: int = 4
    index_path: Path = Path("data/index.json")
    docs_dir: Path = Path("data/docs")

@dataclass
class MCPConfig:
    # 演示：通过子进程/stdio 连接 MCP server（你也可换成 TCP）
    command: list[str] | None = None  # 例如 ["python", "tools/demo_server.py"]
    request_timeout_s: float = 20.0

@dataclass
class AppConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    mcp: MCPConfig = MCPConfig(command=[sys.executable, "tools/math_server.py"])
