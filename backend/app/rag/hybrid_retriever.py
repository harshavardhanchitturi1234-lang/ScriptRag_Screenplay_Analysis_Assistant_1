"""
Hybrid Retriever for Screenwriting Theory Knowledge Base.
Unifies dense vector search, sparse BM25 keyword matching, metadata filtering,
RRF reranking, and citation generation. Explicitly discloses framework limitations.
"""

from typing import List, Dict, Any, Optional
from app.schemas.theory import (
    TheorySearchQuery,
    TheorySearchResult,
    TheorySearchResponse,
    TheoryChunk,
    TheorySourceMetadata
)
from app.rag.vector_store import TheoryVectorStore
from app.rag.bm25_index import BM25Index
from app.rag.reranker import HybridReranker
from app.rag.citation_engine import CitationEngine


class HybridTheoryRetriever:
    """
    Main retrieval orchestrator for screenwriting principles and dramaturgical frameworks.
    """

    def __init__(self, vector_store: Optional[TheoryVectorStore] = None):
        self.vector_store = vector_store or TheoryVectorStore()
        self.bm25_index = BM25Index()
        self.reranker = HybridReranker()
        self.all_chunks: Dict[str, TheoryChunk] = {}
        self.indexed_sources: Dict[str, TheorySourceMetadata] = {}
        self.available_authors: set[str] = set()
        self.available_frameworks: set[str] = set()

    def register_chunks(self, chunks: List[TheoryChunk], sources: Optional[List[TheorySourceMetadata]] = None):
        """Indexes chunks into both the Chroma vector store and the in-memory BM25 index."""
        if not chunks:
            return

        # 1. Update in-memory lookup
        for chunk in chunks:
            self.all_chunks[chunk.chunk_id] = chunk
            self.available_authors.add(chunk.author.lower())

        if sources:
            for s in sources:
                self.indexed_sources[s.source_id] = s
                for fw in s.framework_tags:
                    self.available_frameworks.add(fw.lower())

        # 2. Index into ChromaDB vector store
        self.vector_store.add_chunks(chunks)

        # 3. Index into BM25
        bm25_docs = []
        for chunk in chunks:
            bm25_docs.append({
                "id": chunk.chunk_id,
                "text": chunk.content,
                "concept_name": chunk.concept_name,
                "chapter": chunk.chapter_or_section,
                "metadata": {
                    "source_id": chunk.source_id,
                    "source_title": chunk.source_title,
                    "author": chunk.author,
                    "concept_name": chunk.concept_name,
                    "topic_category": chunk.topic_category,
                    "chapter_or_section": chunk.chapter_or_section,
                    "page_number": chunk.page_number,
                    "publication_info": chunk.publication_info,
                    "license_status": chunk.license_status,
                    "source_url": chunk.source_url,
                    "content": chunk.content,
                    "key_takeaways": chunk.key_takeaways
                }
            })
        self.bm25_index.index_documents(bm25_docs)

    def search(self, request: TheorySearchQuery) -> TheorySearchResponse:
        """
        Executes hybrid retrieval:
        1. Checks for framework/author limitations and discloses if absent.
        2. Queries dense vector database with metadata filters.
        3. Queries sparse BM25 index with metadata filters.
        4. Fuses and reranks results using RRF.
        5. Builds verified TheoryCitation objects.
        """
        disclosed_limitation = None

        # Check if user specifically requested an author or framework not present
        if request.author_filter:
            author_clean = request.author_filter.strip().lower()
            if not any(author_clean in a for a in self.available_authors):
                disclosed_limitation = (
                    f"Author '{request.author_filter}' is not present in the current knowledge base. "
                    f"Available authors: {', '.join(sorted([a.title() for a in self.available_authors]))}."
                )

        if request.framework_filter and not disclosed_limitation:
            fw_clean = request.framework_filter.strip().lower()
            if not any(fw_clean in f for f in self.available_frameworks):
                disclosed_limitation = (
                    f"Framework '{request.framework_filter}' is not indexed. "
                    f"Available frameworks: {', '.join(sorted([f.title() for f in self.available_frameworks]))}."
                )

        # Build ChromaDB metadata filter
        where_clauses = []
        if request.author_filter:
            where_clauses.append({"author": {"$eq": request.author_filter}})
        if request.category_filter:
            where_clauses.append({"topic_category": {"$eq": request.category_filter}})

        where_filter = None
        if len(where_clauses) == 1:
            where_filter = where_clauses[0]
        elif len(where_clauses) > 1:
            where_filter = {"$and": where_clauses}

        # 1. Dense retrieval
        dense_candidates = self.vector_store.search_dense(
            query=request.query,
            top_k=request.top_k * 2,
            where_filter=where_filter
        )

        # 2. Sparse BM25 retrieval
        def bm25_filter(meta: Dict[str, Any]) -> bool:
            if request.author_filter and request.author_filter.lower() not in meta.get("author", "").lower():
                return False
            if request.category_filter and request.category_filter.lower() != meta.get("topic_category", "").lower():
                return False
            if request.concept_filter and request.concept_filter.lower() not in meta.get("concept_name", "").lower():
                return False
            return True

        sparse_candidates = self.bm25_index.search(
            query=request.query,
            top_k=request.top_k * 2,
            filter_fn=bm25_filter
        )

        # 3. Rerank
        reranked = self.reranker.rerank(
            query=request.query,
            dense_results=dense_candidates,
            sparse_results=sparse_candidates,
            top_k=request.top_k
        )

        # 4. Filter by minimum score threshold and construct final response
        results: List[TheorySearchResult] = []
        for item in reranked:
            if item["combined_score"] < request.min_score_threshold:
                continue

            chunk_id = item["chunk_id"]
            chunk = self.all_chunks.get(chunk_id)
            if not chunk:
                chunk = self.vector_store.get_chunk_by_id(chunk_id)

            if chunk:
                citation = CitationEngine.create_citation(
                    chunk=chunk,
                    relevance_score=item["combined_score"]
                )
                results.append(TheorySearchResult(
                    chunk=chunk,
                    dense_score=item["dense_score"],
                    sparse_score=item["sparse_score"],
                    combined_score=item["combined_score"],
                    citation=citation
                ))

        active_frameworks = sorted(list(self.available_frameworks))

        return TheorySearchResponse(
            query=request.query,
            total_matches=len(results),
            results=results,
            available_frameworks_consulted=active_frameworks,
            disclosed_limitations=disclosed_limitation
        )

