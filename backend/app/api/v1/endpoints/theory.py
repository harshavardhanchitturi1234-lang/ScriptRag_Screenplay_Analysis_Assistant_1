"""
Theory Retrieval & Ingestion API Endpoints.
FastAPI v1 endpoints for searching screenwriting knowledge, retrieving citations,
and ingesting user-uploaded reference documents.
"""

import os
import shutil
from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends, status

from app.schemas.theory import (
    TheorySearchQuery,
    TheorySearchResponse,
    TheorySourceMetadata,
    TheoryChunk,
    IngestDocumentRequest
)
from app.rag.hybrid_retriever import HybridTheoryRetriever
from app.rag.vector_store import TheoryVectorStore
from app.rag.extractors import DocumentExtractor
from app.rag.semantic_chunker import SemanticChunker

router = APIRouter(prefix="/theory", tags=["Screenwriting Knowledge Base"])

# Singleton retriever instance initialized from persisted store
_retriever_instance: Optional[HybridTheoryRetriever] = None


def get_retriever() -> HybridTheoryRetriever:
    global _retriever_instance
    if _retriever_instance is None:
        vector_store = TheoryVectorStore()
        retriever = HybridTheoryRetriever(vector_store=vector_store)

        # Locate manifest.json dynamically
        manifest_path = None
        for parent in Path(__file__).resolve().parents:
            cand = parent / "knowledge_base" / "embeddings_index" / "manifest.json"
            if cand.exists():
                manifest_path = cand
                break
        
        if manifest_path and manifest_path.exists():
            import json
            with open(manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            sources = [TheorySourceMetadata(**s) for s in data.get("sources", [])]
            chunks = [TheoryChunk(**c) for c in data.get("chunks", [])]
            retriever.register_chunks(chunks, sources)

        _retriever_instance = retriever
    return _retriever_instance


@router.post("/search", response_model=TheorySearchResponse, summary="Hybrid search screenwriting principles")
def search_theory(
    query_payload: TheorySearchQuery,
    retriever: HybridTheoryRetriever = Depends(get_retriever)
) -> TheorySearchResponse:
    """
    Executes hybrid (dense semantic + sparse BM25) search against screenwriting knowledge base.
    Returns ranked chunks with verified source citations.
    """
    if not query_payload.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Search query cannot be empty."
        )
    return retriever.search(query_payload)


@router.get("/sources", response_model=List[TheorySourceMetadata], summary="List all indexed canonical works")
def list_theory_sources(
    retriever: HybridTheoryRetriever = Depends(get_retriever)
) -> List[TheorySourceMetadata]:
    """
    Returns metadata for all indexed screenwriting books, essays, and educational sources.
    """
    return list(retriever.indexed_sources.values())


@router.post("/ingest", summary="Ingest user-uploaded reference document (PDF, DOCX, TXT)")
async def ingest_reference_document(
    title: str = Form(..., description="Document title"),
    author: str = Form(..., description="Document author"),
    license_status: str = Form("User-Provided Reference Material", description="Usage / license declaration"),
    framework_tags: Optional[str] = Form("", description="Comma-separated framework tags"),
    file: UploadFile = File(..., description="Document file (.pdf, .docx, .txt, .md)"),
    retriever: HybridTheoryRetriever = Depends(get_retriever)
):
    """
    Ingests and indexes a user-provided screenwriting reference document into the knowledge base.
    """
    allowed_exts = {".pdf", ".docx", ".txt", ".md", ".fountain"}
    file_ext = Path(file.filename).suffix.lower() if file.filename else ""
    if file_ext not in allowed_exts:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{file_ext}'. Allowed: {', '.join(allowed_exts)}"
        )

    # Save temporary file
    temp_dir = Path("scratch/uploads")
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / file.filename

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # 1. Extract document
        extracted = DocumentExtractor.extract_file(temp_path)

        # 2. Build metadata
        tags = [t.strip() for t in framework_tags.split(",") if t.strip()] if framework_tags else ["User Reference"]
        source_id = f"user_{Path(file.filename).stem.lower().replace(' ', '_')}"
        source_meta = TheorySourceMetadata(
            source_id=source_id,
            title=title,
            author=author,
            license_status=license_status,
            framework_tags=tags,
            publisher="User Upload"
        )

        # 3. Chunk text semantically
        chunker = SemanticChunker()
        chunks = chunker.chunk_document(extracted.raw_text, source_meta)

        # 4. Register in retriever and vector store
        retriever.register_chunks(chunks, [source_meta])

        return {
            "status": "SUCCESS",
            "message": f"Successfully ingested '{title}' by {author}",
            "source_id": source_id,
            "total_pages": extracted.total_pages,
            "total_words": extracted.total_words,
            "chunks_created": len(chunks)
        }
    finally:
        if temp_path.exists():
            os.remove(temp_path)

