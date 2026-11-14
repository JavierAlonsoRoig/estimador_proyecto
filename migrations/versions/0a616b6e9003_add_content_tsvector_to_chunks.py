"""add content tsvector to chunks

Revision ID: 0a616b6e9003
Revises: 1bc23d6377c8
Create Date: 2026-11-14

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = '0a616b6e9003'
down_revision: Union[str, Sequence[str], None] = '1bc23d6377c8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add a generated tsvector column over chunk content, with a GIN index."""
    # 'english': chunk content (budget components) is written in English.
    # The regconfig must be a literal so the expression is immutable, as GENERATED requires.
    # STORED generated columns are computed for existing rows too: no re-ingestion needed.
    op.add_column(
        "chunks",
        sa.Column(
            "content_tsv",
            postgresql.TSVECTOR(),
            sa.Computed("to_tsvector('english', content)", persisted=True),
            nullable=True,
        ),
    )
    op.create_index(
        "ix_chunks_content_tsv",
        "chunks",
        ["content_tsv"],
        unique=False,
        postgresql_using="gin",
    )


def downgrade() -> None:
    """Drop the full-text search column and its index."""
    op.drop_index("ix_chunks_content_tsv", table_name="chunks", postgresql_using="gin")
    op.drop_column("chunks", "content_tsv")
