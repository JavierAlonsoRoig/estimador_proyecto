import tiktoken
from app.embedding_pipeline.schemas import ChunkMetadata, Chunk


class JSONStructuralChunker:
    """Chunks a budget JSON document at the component level.

    Each budget component becomes one chunk. The chunk text combines
    the component's own fields with contextual headers from the parent
    budget (client sector, year, main technology, project summary).
    """

    def __init__(self, model_for_token_count: str = "text-embedding-3-small"):
        self._tokenizer = tiktoken.encoding_for_model(model_for_token_count)

    def chunk(self, budgets: list[dict]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for budget in budgets:
            chunks.extend(self._chunk_one_budget(budget))
        return chunks

    def _chunk_one_budget(self, budget: dict) -> list[Chunk]:
        parent_context = self._build_parent_context(budget)
        return [
            self._build_chunk(component, budget, parent_context)
            for component in budget["components"]
        ]

    def _build_parent_context(self, budget: dict) -> str:
        client = budget["client_metadata"]
        return (
            f"[Project: {budget['project_summary']}]\\n"
            f"[Client sector: {client['sector']} | "
            f"Year: {budget['year']} | "
            f"Main tech: {budget['main_technology']}]"
        )

    def _build_chunk(
        self, component: dict, budget: dict, parent_context: str
    ) -> Chunk:
        text = self._render_component_text(component, parent_context)
        return Chunk(
            chunk_id=f"{budget['budget_id']}::{component['component_id']}",
            text=text,
            metadata=self._build_metadata(component, budget),
            token_count=len(self._tokenizer.encode(text)),
        )

    def _render_component_text(
        self, component: dict, parent_context: str
    ) -> str:
        return (
            f"{parent_context}\\n\\n"
            f"Component: {component['name']}\\n"
            f"Description: {component['description']}\\n"
            f"Tech stack: {', '.join(component['tech_stack'])}\\n"
            f"Complexity: {component['complexity']}\\n"
            f"Estimated hours: {component['estimated_hours']}"
        )


    def _build_metadata(self, component: dict, budget: dict) -> ChunkMetadata:
        return ChunkMetadata(
            budget_id=budget["budget_id"],
            component_id=component["component_id"],
            client_sector=budget["client_metadata"]["sector"],
            main_technology=budget["main_technology"],
            year=budget["year"],
            complexity=component["complexity"],
            estimated_hours=component["estimated_hours"],
        )