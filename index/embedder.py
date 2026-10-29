"""Local sentence-transformer embedder for Chroma.

Query and document encoding are separate because several models (multi-qa,
bge) were trained with a query-side prefix and none on documents. MiniLM-L6
has no prefix, so today the split is a no-op -- it exists so swapping models
on Day 11 is a config change, not a refactor.
"""

from __future__ import annotations

import os

from chromadb import Documents, EmbeddingFunction, Embeddings
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Prefix required on the query side only, per model. Missing from this dict
# means the model takes no prefix.
QUERY_PREFIXES: dict[str, str] = {
    "sentence-transformers/multi-qa-MiniLM-L6-cos-v1": (
        "Represent this sentence for searching relevant passages: "
    ),
    "BAAI/bge-small-en-v1.5": (
        "Represent this sentence for searching relevant passages: "
    ),
}


class LocalEmbedder(EmbeddingFunction[Documents]):
    """Embeds documents and queries with a local sentence-transformers model."""

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or os.environ.get("EMBED_MODEL", DEFAULT_MODEL)
        self.model = SentenceTransformer(self.model_name)
        self.query_prefix = QUERY_PREFIXES.get(self.model_name, "")

    def encode_documents(self, docs: list[str]) -> list[list[float]]:
        return self.model.encode(
            docs, normalize_embeddings=True, show_progress_bar=False
        ).tolist()

    def encode_query(self, query: str) -> list[float]:
        text = self.query_prefix + query
        return self.model.encode(
            [text], normalize_embeddings=True, show_progress_bar=False
        )[0].tolist()

    def name(self) -> str:
        return self.model_name

    def __call__(self, input: Documents) -> Embeddings:
        # Chroma routes both .add() and .query() here. Documents take no prefix;
        # the retrieval tool passes query vectors explicitly via encode_query.
        return self.encode_documents(list(input))
