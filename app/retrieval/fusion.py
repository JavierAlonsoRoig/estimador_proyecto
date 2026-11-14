from dataclasses import replace

from app.retrieval.types import RetrievedChunk

# k=60 is the value from the original RRF paper (Cormack et al., 2009) and the usual default.
DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    rankings: list[list[RetrievedChunk]], k: int = DEFAULT_RRF_K
) -> list[RetrievedChunk]:
    """Fuse several rankings into one: score(d) = sum over rankings of 1 / (k + rank(d)).

    Only positions matter, not the original scores, so cosine similarity and ts_rank
    (which live on unrelated scales) can be combined without any normalisation.
    """
    scores: dict[int, float] = {}
    chunks: dict[int, RetrievedChunk] = {}
    for ranking in rankings:
        for rank, chunk in enumerate(ranking, start=1):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank)
            chunks.setdefault(chunk.chunk_id, chunk)

    # Tie-break by chunk_id so the fused order is deterministic
    ordered_ids = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))
    return [replace(chunks[chunk_id], score=scores[chunk_id]) for chunk_id in ordered_ids]
