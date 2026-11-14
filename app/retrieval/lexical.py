from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.retrieval.types import RetrievedChunk

# plainto_tsquery joins every term with AND (&). A project description has many terms and
# almost no chunk contains all of them, so we rewrite the query as OR (|) and let ts_rank
# reward chunks that match more (and rarer) terms.
# 'english' must match the config of chunks.content_tsv, or the stems won't line up.
LEXICAL_SEARCH_SQL = text(
    """
    WITH q AS (
        SELECT to_tsquery(
            'english',
            replace(plainto_tsquery('english', :query)::text, ' & ', ' | ')
        ) AS tsq
    )
    SELECT c.id, c.document_id, c.chunk_type, c.content, c.metadata,
           ts_rank(c.content_tsv, q.tsq) AS rank
    FROM chunks c, q
    WHERE c.content_tsv @@ q.tsq
    ORDER BY rank DESC, c.id
    LIMIT :limit
    """
)


async def lexical_search(session: AsyncSession, query: str, limit: int) -> list[RetrievedChunk]:
    """Chunks matching the query terms (full-text search), best first."""
    rows = (await session.execute(LEXICAL_SEARCH_SQL, {"query": query, "limit": limit})).all()
    return [
        RetrievedChunk(
            chunk_id=row.id,
            document_id=row.document_id,
            chunk_type=row.chunk_type,
            content=row.content,
            metadata=row.metadata,
            score=float(row.rank),
        )
        for row in rows
    ]
