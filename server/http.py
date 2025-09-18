from fastapi import FastAPI
from pydantic import BaseModel
from retriever.pipeline import reindex as do_reindex, query as do_query

app = FastAPI()

class ChatReq(BaseModel):
    mode: str = "chat"
    text: str

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/api/reindex")
def api_reindex():
    stats = do_reindex()
    return {"ok": True, "stats": stats}

class QueryReq(BaseModel):
    q: str
    top_k: int | None = None

@app.post("/api/query")
def api_query(req: QueryReq):
    hits = do_query(req.q, req.top_k)
    return {"hits": hits}
