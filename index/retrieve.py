"""Corpus retrieval, clause lookup, and request field extraction.

Connects runtime retrieval to the Chroma vector index (L1, F5).
"""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import chromadb

from contracts import Chunk, ProvisionStatus
from index.embedder import LocalEmbedder

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = ROOT / "index" / "store"
COLLECTION = "corpus"

# Standard statutory provisions for the workplace compliance corpus
# Kept as offline fallback when the Chroma index store has not been generated.
CORPUS_CHUNKS: list[Chunk] = [
    # Code on Social Security 2020 - Section 53
    Chunk(
        chunk_id="css2020-s53-main",
        doc="css2020",
        section="53",
        part="main",
        status=ProvisionStatus.IN_FORCE,
        text=(
            "53.(1) Gratuity shall be payable to an employee on the termination of his employment "
            "after he has rendered continuous service for not less than five years: "
            "(a) on his superannuation; or (b) on his retirement or resignation; or "
            "(c) on his death or disablement due to accident or disease."
        ),
    ),
    Chunk(
        chunk_id="css2020-s53-proviso_1",
        doc="css2020",
        section="53",
        part="proviso_1",
        status=ProvisionStatus.IN_FORCE,
        text=(
            "Provided that in case of working journalist as defined in clause (f) of section 2 of the "
            "Working Journalists and Other Newspaper Employees (Condition of Service) and "
            "Miscellaneous Provisions Act, 1955, the expression 'five years' occurring in this "
            "sub-section shall be deemed to be three years."
        ),
    ),
    Chunk(
        chunk_id="css2020-s53-proviso_2",
        doc="css2020",
        section="53",
        part="proviso_2",
        status=ProvisionStatus.IN_FORCE,
        text=(
            "Provided further that the completion of continuous service of five years shall not be "
            "necessary where the termination of the employment of any employee is due to death or "
            "disablement or expiration of fixed term employment or happening of any such event as "
            "may be notified by the Central Government."
        ),
    ),
    Chunk(
        chunk_id="css2020-s53-proviso_3",
        doc="css2020",
        section="53",
        part="proviso_3",
        status=ProvisionStatus.IN_FORCE,
        text=(
            "Provided also that in the case of death of the employee, gratuity payable to him "
            "shall be paid to his nominee or, if no nomination has been made, to his heirs."
        ),
    ),
    Chunk(
        chunk_id="css2020-s53-proviso_4",
        doc="css2020",
        section="53",
        part="proviso_4",
        status=ProvisionStatus.IN_FORCE,
        text=(
            "Provided that in the case of a piece-rated employee, daily wages shall be computed "
            "on the average of the total wages received by him for a period of three months."
        ),
    ),
    Chunk(
        chunk_id="css2020-s53-proviso_5",
        doc="css2020",
        section="53",
        part="proviso_5",
        status=ProvisionStatus.IN_FORCE,
        text=(
            "Provided further that in the case of an employee who is employed in a seasonal "
            "establishment and who is not so employed throughout the year, the employer shall pay the "
            "gratuity at the rate of seven days' wages for each season."
        ),
    ),
    Chunk(
        chunk_id="css2020-s53-proviso_6",
        doc="css2020",
        section="53",
        part="proviso_6",
        status=ProvisionStatus.IN_FORCE,
        text=(
            "Provided also that in the case of an employee employed on fixed term employment or "
            "a deceased employee, the employer shall pay gratuity on pro rata basis."
        ),
    ),
    # Payment of Gratuity Act 1972 (Repealed) - Section 4
    Chunk(
        chunk_id="pga1972-s4-main",
        doc="pga1972",
        section="4",
        part="main",
        status=ProvisionStatus.REPEALED,
        text=(
            "4.(1) Gratuity shall be payable to an employee on the termination of his employment "
            "after he has rendered continuous service for not less than five years "
            "(a) on his superannuation, or (b) on his retirement or resignation, or "
            "(c) on his death or disablement due to accident or disease."
        ),
    ),
]

_client: chromadb.PersistentClient | None = None
_collection: chromadb.Collection | None = None
_embedder: LocalEmbedder | None = None


def _get_collection() -> chromadb.Collection | None:
    """Lazy initialize and return the persistent Chroma collection."""
    global _client, _collection, _embedder
    if _collection is not None:
        return _collection
    if not INDEX_DIR.exists():
        return None
    try:
        if _client is None:
            _client = chromadb.PersistentClient(path=str(INDEX_DIR))
        existing = [c.name for c in _client.list_collections()]
        if COLLECTION not in existing:
            return None
        if _embedder is None:
            _embedder = LocalEmbedder()
        _collection = _client.get_collection(COLLECTION, embedding_function=_embedder)
        return _collection
    except Exception as exc:  # noqa: BLE001
        logger.warning("Chroma collection load failed; using in-memory corpus: %s", exc)
        return None


def _sort_chunks(chunks: list[Chunk]) -> list[Chunk]:
    """Sort chunks into canonical statutory document order."""

    def key(chunk: Chunk) -> tuple[int, int]:
        part = chunk.part
        if part == "main":
            return (0, 0)
        if part.startswith("proviso_"):
            try:
                return (1, int(part.split("_")[1]))
            except ValueError:
                return (1, 99)
        if part.startswith("explanation"):
            try:
                num = int(part.split("_")[1]) if "_" in part else 1
                return (2, num)
            except ValueError:
                return (2, 99)
        return (3, 0)

    return sorted(chunks, key=key)


def get_index_metadata() -> dict[str, Any]:
    """Return build metadata stored in Chroma for observability (/metrics) (D2)."""
    col = _get_collection()
    if col is None:
        return {
            "chunk_count": len(CORPUS_CHUNKS),
            "freshness_seconds": 0.0,
            "status": "fallback",
            "corpus_hash": "unindexed",
            "embed_model": "none",
            "built_at": None,
        }

    meta = col.metadata or {}
    built_at_str = meta.get("built_at")
    freshness_s = 0.0
    if built_at_str:
        try:
            built_dt = datetime.fromisoformat(built_at_str)
            freshness_s = max(0.0, (datetime.now(UTC) - built_dt).total_seconds())
        except Exception:
            pass

    return {
        "chunk_count": col.count(),
        "freshness_seconds": round(freshness_s, 1),
        "status": "ready",
        "corpus_hash": str(meta.get("corpus_hash", "unknown")),
        "embed_model": str(meta.get("embed_model", "unknown")),
        "built_at": built_at_str,
    }


def lookup_clause(
    section: str,
    doc: str = "css2020",
    part_prefix: str | None = None,
) -> list[Chunk]:
    """Fetch all parts of a section by number in document order."""
    clean_sec = section.strip()
    match = re.match(r"^(\d+)", clean_sec)
    sec_num = match.group(1) if match else clean_sec

    col = _get_collection()
    if col is not None:
        try:
            records = col.get(
                where={"$and": [{"doc": doc}, {"section": sec_num}]}
            )
            chunks: list[Chunk] = []
            if records and records.get("ids"):
                for cid, meta, doc_text in zip(
                    records["ids"], records["metadatas"], records["documents"]
                ):
                    if part_prefix is not None and not meta["part"].startswith(part_prefix):
                        continue
                    chunks.append(
                        Chunk(
                            chunk_id=cid,
                            doc=meta["doc"],
                            section=meta["section"],
                            part=meta["part"],
                            status=ProvisionStatus(meta["status"]),
                            text=doc_text,
                        )
                    )
            if chunks:
                return _sort_chunks(chunks)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Chroma lookup_clause failed, falling back: %s", exc)

    # In-memory fallback
    results: list[Chunk] = []
    for c in CORPUS_CHUNKS:
        if c.doc == doc and (c.section == sec_num or c.section.startswith(sec_num)):
            if part_prefix is not None and not c.part.startswith(part_prefix):
                continue
            results.append(c)
    return _sort_chunks(results)


def corpus_search(
    query: str,
    status: ProvisionStatus | None = None,
    doc: str | None = None,
    section: str | None = None,
    k: int = 5,
) -> list[Chunk]:
    """Semantic vector search over the Chroma corpus (L1/F5).

    Queries Chroma embeddings without hand-tuned scoring heuristics.
    """
    col = _get_collection()
    if col is not None:
        conditions: list[dict[str, Any]] = []
        if status is not None:
            conditions.append({"status": status.value})
        if doc is not None:
            conditions.append({"doc": doc})
        if section is not None:
            conditions.append({"section": section})

        where: dict[str, Any] | None = None
        if len(conditions) == 1:
            where = conditions[0]
        elif len(conditions) > 1:
            where = {"$and": conditions}

        try:
            results = col.query(query_texts=[query], n_results=k, where=where)
            chunks: list[Chunk] = []
            if results and results.get("ids") and results["ids"][0]:
                for cid, meta, doc_text in zip(
                    results["ids"][0], results["metadatas"][0], results["documents"][0]
                ):
                    chunks.append(
                        Chunk(
                            chunk_id=cid,
                            doc=meta["doc"],
                            section=meta["section"],
                            part=meta["part"],
                            status=ProvisionStatus(meta["status"]),
                            text=doc_text,
                        )
                    )
                return chunks
        except Exception as exc:  # noqa: BLE001
            logger.warning("Chroma vector query failed, falling back: %s", exc)

    # In-memory fallback
    candidates = list(CORPUS_CHUNKS)
    if status is not None:
        candidates = [c for c in candidates if c.status == status]
    if doc is not None:
        candidates = [c for c in candidates if c.doc == doc]
    if section is not None:
        candidates = [c for c in candidates if c.section == section]

    return candidates[:k]


def lookup_request_field(field: str, request: dict[str, Any]) -> Any:
    """Read one field from the employer's request by dotted path."""
    parts = field.split(".")
    curr: Any = request
    for part in parts:
        if isinstance(curr, dict) and part in curr:
            curr = curr[part]
        else:
            return None
    return curr
