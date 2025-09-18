import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
    INDEX_DIR = os.getenv("INDEX_DIR", ".data/index")
    DOCS_DIR = os.getenv("DOCS_DIR", "./data/docs")
    EMBED_DIM = int(os.getenv("EMBED_DIM", "384"))
    TOP_K = int(os.getenv("TOP_K", "5"))
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "700"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))
    RUNNER_TIMEOUT_SEC = int(os.getenv("RUNNER_TIMEOUT_SEC", "6"))
    MCP_BINARIES = os.getenv("MCP_BINARIES", "")

settings = Settings()
