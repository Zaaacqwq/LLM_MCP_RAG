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

# ---------- LLM ----------
@dataclass
class LLMConfig:
    provider: str = "openai"
    model: str = "gpt-4.1"
    api_base: str = "https://api.openai.com/v1"
    api_key: str = OPENAI_API_KEY

# ---------- RAG ----------
@dataclass
class RAGConfig:
    chunk_size: int = 600      # 默认 chunk 大小（不一定用）
    chunk_overlap: int = 120   # 默认 overlap
    top_k: int = 4
    index_path: Path = Path("data/index.json")
    docs_dir: Path = Path("data/docs")

# ---------- MCP ----------
@dataclass
class MCPConfig:
    # 既支持单个命令 list[str]，也支持多个命令 list[list[str]]
    command: list | None = None
    request_timeout_s: float = 20.0

# ---------- Chunk ----------
@dataclass
class ChunkConfig:
    # 可选: "llm_outline" | "semantic_ai" | "recursive" | "fixed" | "sentence" | "heading"
    name: str = "llm_outline"
    chunk_size: int = 1200
    chunk_overlap: int = 150
    semantic_model: str = "all-MiniLM-L6-v2"
    semantic_sim_threshold: float = 0.55
    max_chars_per_call: int = 18000  # 仅 llm_outline 用

# ---------- AppConfig ----------
@dataclass
class AppConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    mcp: MCPConfig = MCPConfig(command=[
        [sys.executable, "tools/math_server.py"],
        [sys.executable, "tools/code_server.py"],
        [sys.executable, "tools/file_server.py"],
    ])
    chunk: ChunkConfig = field(default_factory=ChunkConfig)

# 全局单例
APP = AppConfig()
CHUNK = APP.chunk
