from fastapi import FastAPI
from app.routers import estimations, sessions
from app.embedding_pipeline.router import router as embeddings_router
from app.embedding_pipeline.search import router as search_router
from app.generation.router import router as rag_router
from uuid import uuid4

import structlog
from fastapi import Request


app = FastAPI(
    title="Estimador CAG",
    description="Sistema de estimación de software con arquitectura CAG",
    version="0.1.0"
)
@app.middleware("http")
async def bind_request_id(request: Request, call_next):
    # Every log line emitted while serving this request carries the same request_id
    request_id = request.headers.get("X-Request-ID") or uuid4().hex
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(request_id=request_id)
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


app.include_router(estimations.router)
app.include_router(sessions.router)
app.include_router(embeddings_router, prefix="/embeddings", tags=["embeddings"])
app.include_router(search_router, tags=["embeddings"])

app.include_router(rag_router, prefix="/rag", tags=["rag"])


@app.get("/health")
async def health():
    return {"status": "healthy"}

