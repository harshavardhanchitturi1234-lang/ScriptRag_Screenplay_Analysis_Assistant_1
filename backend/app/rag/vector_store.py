"""
Vector Database Adapter (ChromaDB) for Screenwriting Knowledge Base.
Provides persistent storage, metadata filtering, and dense cosine similarity search.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings

from app.schemas.theory import TheoryChunk
from app.rag.embeddings import TheoryEmbeddingEngine


class TheoryVectorStore:
    """Manages persistent ChromaDB collections for screenwriting theory and references."""

    COLLECTION_NAME = "screenwriting_theory_kb"

    def __init__(self, persist_directory: Optional[str] = None):
        if persist_directory is None:
            # Default to knowledge_base/embeddings_index/chroma in project root
            base_dir = Path(__file__).resolve().parent.parent.parent.parent
            self.persist_directory = str(base_dir / "knowledge_base" / "embeddings_index" / "chroma")
        else:
            self.persist_directory = persist_directory

        os.makedirs(self.persist_directory, exist_ok=True)
        self.embedding_engine = TheoryEmbeddingEngine()

        # Initialize Chroma persistent client
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"description": "Canonical Filmmaking & Screenwriting Theory Corpus"}
        )

    def add_chunks(self, chunks: List[TheoryChunk]):
        """Adds or updates chunks in the vector collection."""
        if not chunks:
            return

        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.content for chunk in chunks]
        
        # Metadata must have scalar values for ChromaDB
        metadatas = []
        for chunk in chunks:
            meta = {
                "source_id": chunk.source_id,
                "source_title": chunk.source_title,
                "author": chunk.author,
                "chapter_or_section": chunk.chapter_or_section,
                "page_number": chunk.page_number if chunk.page_number is not None else -1,
                "concept_name": chunk.concept_name,
                "topic_category": chunk.topic_category,
                "license_status": chunk.license_status,
                "publication_info": chunk.publication_info,
                "source_url": chunk.source_url or "",
                "key_takeaways": " || ".join(chunk.key_takeaways)
            }
            metadatas.append(meta)

        embeddings = self.embedding_engine.embed_texts(documents)

        # Upsert into ChromaDB
        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )

    def search_dense(
        self,
        query: str,
        top_k: int = 10,
        where_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes dense vector search using cosine distance in ChromaDB.
        Returns list of matched dicts with chunk_id, score, document, and metadata.
        """
        count = self.collection.count()
        if count == 0:
            return []

        query_embedding = self.embedding_engine.embed_query(query)
        actual_k = min(top_k, count)

        kwargs: Dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": actual_k,
            "include": ["documents", "metadatas", "distances"]
        }
        if where_filter:
            kwargs["where"] = where_filter

        results = self.collection.query(**kwargs)

        matched_results = []
        if results and "ids" in results and results["ids"]:
            ids = results["ids"][0]
            docs = results["documents"][0] if "documents" in results else []
            metas = results["metadatas"][0] if "metadatas" in results else []
            distances = results["distances"][0] if "distances" in results else []

            for idx, chunk_id in enumerate(ids):
                dist = distances[idx] if idx < len(distances) else 1.0
                # Chroma cosine distance ranges from 0 (identical) to 2.
                # Convert distance to normalized similarity score (0.0 to 1.0)
                similarity = max(0.0, min(1.0, 1.0 - (dist / 2.0)))

                matched_results.append({
                    "chunk_id": chunk_id,
                    "content": docs[idx] if idx < len(docs) else "",
                    "metadata": metas[idx] if idx < len(metas) else {},
                    "dense_score": round(similarity, 4)
                })

        return matched_results

    def get_chunk_by_id(self, chunk_id: str) -> Optional[TheoryChunk]:
        """Fetches a specific chunk by its ID."""
        res = self.collection.get(ids=[chunk_id], include=["documents", "metadatas"])
        if not res or not res["ids"]:
            return None

        doc = res["documents"][0]
        meta = res["metadatas"][0]
        takeaways = meta.get("key_takeaways", "").split(" || ") if meta.get("key_takeaways") else []
        page_num = meta.get("page_number")
        if page_num == -1:
            page_num = None

        return TheoryChunk(
            chunk_id=chunk_id,
            source_id=meta.get("source_id", ""),
            source_title=meta.get("source_title", ""),
            author=meta.get("author", ""),
            chapter_or_section=meta.get("chapter_or_section", ""),
            page_number=page_num,
            concept_name=meta.get("concept_name", ""),
            topic_category=meta.get("topic_category", "structure"),
            content=doc,
            key_takeaways=takeaways,
            license_status=meta.get("license_status", ""),
            publication_info=meta.get("publication_info", ""),
            source_url=meta.get("source_url") or None
        )

    def count(self) -> int:
        """Returns total chunks stored in the collection."""
        return self.collection.count()

