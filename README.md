# LLM\_MCP\_RAG

A modular framework for building LLM-powered applications with MCP tools and Retrieval-Augmented Generation (RAG).
This project combines a CLI, HTTP server, retrieval pipeline, and sandboxed tools into a unified orchestration layer.

## Features

* LLM integration: simple wrapper for chat models (OpenAI or others).
* Retrieval-Augmented Generation (RAG):

  * Document loading, chunking, embedding, and HNSW indexing.
  * End-to-end reindex and query pipeline.
* MCP Tools: execute code, solve math, or read files in isolated subprocesses.
* CLI and HTTP: unified orchestrator (Orchestrator) drives both command line and API modes.
* Configurable: runtime settings managed via .env.
* Testing: unit and integration tests with pytest.

## Project Structure

agent/        CLI, Orchestrator, LLM wrapper, Memory
config/       Settings loader (.env)
retriever/    RAG pipeline (reader, chunker, embedder, indexer, store)
server/       HTTP API (FastAPI)
tools/        MCP tool servers (code, math, file) and tool router
scripts/      Utility scripts (e.g., reindex)
tests/        Unit and integration tests
.data/        Local index and cache (ignored by Git)

## Getting Started

1. Clone and install
   git clone <your-repo-url>
   cd LLM\_MCP\_RAG
   python -m venv .venv
   source .venv/bin/activate   (Windows: .venv\Scripts\activate)
   pip install -U pip setuptools wheel
   pip install -e .

2. Configure
   Copy .env.example to .env and set:

   * OPENAI\_API\_KEY=sk-...
   * MODEL\_NAME=gpt-4o-mini
   * DOCS\_DIR=.data/docs
   * INDEX\_DIR=.data/index

3. Rebuild index
   python -m agent.cli /reindex

4. Run CLI
   python -m agent.cli
   Example commands:
   /explain Bayes theorem
   /solve x^2+3x+2=0
   /run python print("Hello MCP")

5. Run HTTP server
   uvicorn server.http\:app --reload
   Visit [http://localhost:8000/docs](http://localhost:8000/docs) for API documentation.

## Development

* Reindex only: ./scripts/dev\_reindex.sh
* Run tests: pytest -q

## Packaging

Build source and wheel distributions:
python -m build
Results are written to dist/

## Extending

* Add a new tool: create tools/mcp/servers/\<new\_server>.py and register it in tools/tool\_router.py
* Customize chunking: edit retriever/chunker.py
* Swap embedding model: update retriever/embedder.py
* Modify prompt orchestration: adjust agent/orchestrator.py

---