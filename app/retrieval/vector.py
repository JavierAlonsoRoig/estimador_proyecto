from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Chunk
from app.retrieval.types import RetrievedChunk


async def vector_search(
    session: AsyncSession, query_vector: list[float], limit: int
) -> list[RetrievedChunk]:
    """Nearest chunks by cosine distance, best first."""
    distance = Chunk.embedding.cosine_distance(query_vector)
    stmt = (
        select(
            Chunk.id,
            Chunk.document_id,
            Chunk.chunk_type,
            Chunk.content,
            Chunk.metadata_,
            distance.label("distance"),
        )
        # embedding is nullable: chunks not embedded yet can't be ranked
        .where(Chunk.embedding.is_not(None))
        .order_by(distance)
        .limit(limit)
    )
    rows = (await session.execute(stmt)).all()
    return [
        RetrievedChunk(
            chunk_id=row.id,
            document_id=row.document_id,
            chunk_type=row.chunk_type,
            content=row.content,
            metadata=row.metadata_,
            # similarity = 1 - distance, so that every branch scores "higher is better"
            score=1.0 - row.distance,
        )
        for row in rows
    ]
