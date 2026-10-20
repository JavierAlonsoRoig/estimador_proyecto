from fastapi import FastAPI
from app.routers import estimations, sessions
from app.embedding_pipeline.router import router as embeddings_router


app = FastAPI(
    title="Estimador CAG",
    description="Sistema de estimación de software con arquitectura CAG",
    version="0.1.0"
)

app.include_router(estimations.router)
app.include_router(sessions.router)
app.include_router(embeddings_router, prefix="/embeddings", tags=["embeddings"])

@app.get("/health")
async def health():
    return {"status": "healthy"}

