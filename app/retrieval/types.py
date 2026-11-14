from dataclasses import dataclass, field


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk returned by any retrieval branch.

    score is always "higher is better", but its scale depends on who produced it:
    cosine similarity (vector), ts_rank (lexical), RRF score (hybrid) or cross-encoder logit (rerank).
    """

    chunk_id: int
    document_id: int
    chunk_type: str
    content: str
    metadata: dict = field(default_factory=dict)
    score: float = 0.0
