"""Build the Chroma index from the corpus PDFs.

Idempotent (D2): running it twice produces the same collection. The collection
records which embedding model built it, so querying with a different model
raises instead of returning plausible noise.

Usage:
    python -m index.build              # rebuild if stale
    python -m index.build --force      # drop and re-embed
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import chromadb

from contracts.analyst_result import Chunk, ProvisionStatus
from index.chunker import chunk_document
from index.embedder import LocalEmbedder

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "corpus"
INDEX_DIR = ROOT / "index" / "store"
COLLECTION = "corpus"

# Repealed acts sit in the corpus deliberately: they are what most material
# online still describes, and the §4.1 trap needs the contrast.
DOCS: list[tuple[str, str, ProvisionStatus]] = [
    ("css2020", "the_code_on_social_security_2020.pdf", ProvisionStatus.IN_FORCE),
    ("posh", "doe_prevention_sexual_harassment_2013.pdf", ProvisionStatus.IN_FORCE),
    ("pga1972", "the_payment_of_gratuity_act_1972.pdf", ProvisionStatus.REPEALED),
]


def pdf_to_text(pdf: Path) -> str:
    """Extract layout-preserving text. Layout mode keeps section numbers at line start."""
    result = subprocess.run(
        ["pdftotext", "-layout", str(pdf), "-"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def digest(chunks: list[Chunk], texts: dict[str, str]) -> str:
    """Fingerprint the whole build: source text plus the chunks it produced.

    Hashing the chunks covers chunker changes too. A build that produced the
    same chunks is genuinely the same index; one that produced different ids
    or text must be rebuilt.
    """
    h = hashlib.sha256()
    for name in sorted(texts):
        h.update(name.encode())
        h.update(texts[name].encode())
    for c in chunks:
        h.update(c.chunk_id.encode())
        h.update(c.text.encode())
    return h.hexdigest()[:16]


def collect_chunks() -> tuple[list[Chunk], dict[str, str]]:
    texts: dict[str, str] = {}
    chunks: list[Chunk] = []
    for doc, filename, status in DOCS:
        path = CORPUS / filename
        if not path.exists():
            raise FileNotFoundError(f"missing corpus file: {path}")
        text = pdf_to_text(path)
        texts[doc] = text
        doc_chunks = chunk_document(text, doc, status)
        chunks.extend(doc_chunks)
        print(f"  {doc:<8} {len(doc_chunks):>4} chunks  ({status.value})")
    return chunks, texts


def build(force: bool = False) -> chromadb.Collection:
    embedder = LocalEmbedder()
    client = chromadb.PersistentClient(path=str(INDEX_DIR))

    print("extracting and chunking:")
    chunks, texts = collect_chunks()
    corpus_hash = digest(chunks, texts)

    existing = [c.name for c in client.list_collections()]
    if COLLECTION in existing:
        col = client.get_collection(COLLECTION, embedding_function=embedder)
        built_with = col.metadata.get("embed_model")
        if built_with != embedder.model_name:
            raise RuntimeError(
                f"index built with {built_with}, querying with {embedder.model_name}. "
                "Rebuild with --force."
            )
        if not force and col.metadata.get("corpus_hash") == corpus_hash:
            print(f"index up to date ({corpus_hash}), skipping. Use --force to rebuild.")
            return col
        print(f"rebuilding (force={force}, hash {corpus_hash})")
        client.delete_collection(COLLECTION)

    col = client.create_collection(
        COLLECTION,
        embedding_function=embedder,
        metadata={
            "embed_model": embedder.model_name,
            "corpus_hash": corpus_hash,
            "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "chunk_count": len(chunks),
        },
    )

    try:
        print(f"embedding {len(chunks)} chunks with {embedder.model_name}")
        batch = 64
        for i in range(0, len(chunks), batch):
            window = chunks[i : i + batch]
            col.add(
                ids=[c.chunk_id for c in window],
                documents=[c.text for c in window],
                metadatas=[
                    {
                        "doc": c.doc,
                        "section": c.section,
                        "part": c.part,
                        "status": c.status.value,
                    }
                    for c in window
                ],
            )
            print(f"  {min(i + batch, len(chunks)):>4}/{len(chunks)}", end="\r")
        print()

        if col.count() != len(chunks):
            raise RuntimeError(f"count mismatch: {col.count()} != {len(chunks)}")
    except Exception:
        # A half-written collection reports a hash for a build that never
        # finished. Drop it so the next run cannot mistake it for current.
        client.delete_collection(COLLECTION)
        raise

    print(f"done: {col.count()} chunks in {INDEX_DIR}")
    return col


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--force", action="store_true", help="drop and re-embed")
    args = p.parse_args(argv)
    build(force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())