"""
Knowledge Base Indexer CLI & Service.
Loads all canonical screenwriting corpus files, generates semantic chunks with citations,
and indexes them into ChromaDB and the BM25 sparse index.
"""

import os
import sys
import json
from pathlib import Path
from typing import List

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(backend_dir))

from app.schemas.theory import TheorySourceMetadata, TheoryChunk
from app.rag.vector_store import TheoryVectorStore
from app.rag.bm25_index import BM25Index


def load_corpus_files(corpus_dir: Path) -> tuple[List[TheorySourceMetadata], List[TheoryChunk]]:
    """Loads all structured theory JSON corpus files."""
    sources: List[TheorySourceMetadata] = []
    chunks: List[TheoryChunk] = []

    json_files = sorted(corpus_dir.glob("*.json"))
    for f in json_files:
        with open(f, "r", encoding="utf-8") as fp:
            data = json.load(fp)

        source_meta = TheorySourceMetadata(**data["source"])
        sources.append(source_meta)

        for sec in data["sections"]:
            chunk_id = f"{source_meta.source_id}_{sec['concept_name'].lower().replace(' ', '_')[:30]}"
            pub_info = f"{source_meta.author}, '{source_meta.title}' ({source_meta.publisher or 'Academic'}, {source_meta.publication_year or 'N/A'})"

            chunk = TheoryChunk(
                chunk_id=chunk_id,
                source_id=source_meta.source_id,
                source_title=source_meta.title,
                author=source_meta.author,
                chapter_or_section=sec["heading"],
                page_number=sec.get("page_number"),
                concept_name=sec["concept_name"],
                topic_category=sec["topic_category"],
                content=sec["content"],
                key_takeaways=sec.get("key_takeaways", []),
                license_status=source_meta.license_status,
                publication_info=pub_info,
                source_url=source_meta.source_url
            )
            chunks.append(chunk)

    return sources, chunks


def index_knowledge_base():
    """Executes the full indexing process."""
    base_dir = Path(__file__).resolve().parent
    corpus_dir = base_dir / "theory_corpus"
    index_dir = base_dir / "embeddings_index"
    chroma_dir = index_dir / "chroma"

    print("=" * 60)
    print("STORY ANALYZER AI: KNOWLEDGE BASE INDEXER")
    print("=" * 60)
    print(f"Loading corpus from: {corpus_dir}")

    sources, chunks = load_corpus_files(corpus_dir)
    print(f"Loaded {len(sources)} canonical works containing {len(chunks)} core concept chunks.")

    # Initialize ChromaDB vector store
    print(f"Indexing into ChromaDB at: {chroma_dir}...")
    vector_store = TheoryVectorStore(persist_directory=str(chroma_dir))
    vector_store.add_chunks(chunks)
    print(f"Vector Store count: {vector_store.count()} chunks indexed.")

    # Build BM25 index and serialize for fast loading
    print("Building BM25 sparse index...")
    bm25 = BM25Index()
    bm25_docs = []
    for c in chunks:
        bm25_docs.append({
            "id": c.chunk_id,
            "text": c.content,
            "concept_name": c.concept_name,
            "chapter": c.chapter_or_section,
            "metadata": c.model_dump()
        })
    bm25.index_documents(bm25_docs)

    # Save manifest of indexed works and metadata
    manifest_path = index_dir / "manifest.json"
    manifest = {
        "total_sources": len(sources),
        "total_chunks": len(chunks),
        "sources": [s.model_dump() for s in sources],
        "chunks": [c.model_dump() for c in chunks]
    }
    with open(manifest_path, "w", encoding="utf-8") as fp:
        json.dump(manifest, fp, indent=2)

    print(f"Saved manifest to: {manifest_path}")
    print("Knowledge Base Indexing Complete!")
    print("=" * 60)
    return sources, chunks


if __name__ == "__main__":
    index_knowledge_base()

