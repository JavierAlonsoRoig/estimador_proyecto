from dataclasses import dataclass

from app.retrieval.types import RetrievedChunk


@dataclass(frozen=True)
class AssembledContext:
    text: str
    # What the LLM actually saw, keyed by chunk_id: the ground truth for citation checks
    document_by_chunk_id: dict[str, str]
    content_by_chunk_id: dict[str, str]
    chunks: list[RetrievedChunk]

    @property
    def chunk_ids(self) -> set[str]:
        return set(self.document_by_chunk_id)


def assemble_context(chunks: list[RetrievedChunk]) -> AssembledContext:
    """Render retrieved chunks as identified <source> blocks for the generation prompt."""
    blocks: list[str] = []
    document_by_chunk_id: dict[str, str] = {}
    content_by_chunk_id: dict[str, str] = {}
    for chunk in chunks:
        chunk_id = str(chunk.chunk_id)
        document_id = chunk.metadata["budget_id"]
        # The chunker stored literal "\n" sequences: restore real newlines for the LLM
        content = chunk.content.replace("\\n", "\n")
        document_by_chunk_id[chunk_id] = document_id
        content_by_chunk_id[chunk_id] = content
        blocks.append(
            f'<source chunk_id="{chunk_id}" document_id="{document_id}" '
            f'sector="{chunk.metadata.get("client_sector")}" year="{chunk.metadata.get("year")}">\n'
            f"{content}\n</source>"
        )
    return AssembledContext(
        text="\n\n".join(blocks),
        document_by_chunk_id=document_by_chunk_id,
        content_by_chunk_id=content_by_chunk_id,
        chunks=chunks,
    )
