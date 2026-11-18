"""
Comprehensive Test Suite for Screenwriting Knowledge Base & RAG Pipeline.
Verifies document extraction (DOCX, TXT), semantic chunking, BM25 indexing,
dense vector retrieval, hybrid RRF reranking, citation formatting, and limitation disclosures.
"""

import sys
import os
import json
from pathlib import Path
import pytest

# Ensure backend root is on sys.path
backend_path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_path))

from app.schemas.theory import TheorySearchQuery, TheorySourceMetadata, TheoryChunk
from app.rag.extractors import DocumentExtractor
from app.rag.text_cleaner import TextCleaner
from app.rag.semantic_chunker import SemanticChunker
from app.rag.bm25_index import BM25Index
from app.rag.vector_store import TheoryVectorStore
from app.rag.hybrid_retriever import HybridTheoryRetriever
from app.rag.citation_engine import CitationEngine
from app.api.v1.endpoints.theory import get_retriever, search_theory, list_theory_sources


@pytest.fixture(scope="session")
def retriever():
    """Provides the active retriever loaded with the canonical corpus."""
    return get_retriever()


def test_text_cleaner():
    """Verifies typographic normalization, smart quotes, and line breaks."""
    raw = "He said \u201chello\u201d\u2014it was his \u2018destiny\u2026\u2019\n\n\n\nNew line."
    cleaned = TextCleaner.clean(raw)
    assert '"hello"' in cleaned
    assert "'destiny...'" in cleaned
    assert " -- " in cleaned
    assert "\n\n\n" not in cleaned


def test_docx_and_txt_extraction(tmp_path):
    """Verifies multi-format extraction for TXT and DOCX."""
    # 1. Test TXT extraction
    txt_file = tmp_path / "sample_rule.txt"
    txt_file.write_text("# Chapter 1: The Inciting Incident\n\nA dramatic disturbance occurs.", encoding="utf-8")
    txt_res = DocumentExtractor.extract_file(txt_file)
    assert txt_res.file_type == "TEXT"
    assert "Inciting Incident" in txt_res.raw_text
    assert txt_res.total_words > 0

    # 2. Test DOCX extraction
    import docx
    docx_file = tmp_path / "sample_guide.docx"
    doc = docx.Document()
    doc.add_heading("Three-Act Structure Overview", level=1)
    doc.add_paragraph("Act One establishes the protagonist's status quo.")
    doc.save(str(docx_file))

    docx_res = DocumentExtractor.extract_file(docx_file)
    assert docx_res.file_type == "DOCX"
    assert "Three-Act Structure Overview" in docx_res.raw_text
    assert "protagonist's status quo" in docx_res.raw_text


def test_bm25_exact_screenwriting_terms():
    """Verifies BM25 index precision on distinct technical terms."""
    bm25 = BM25Index()
    docs = [
        {"id": "doc1", "concept_name": "All Is Lost", "chapter": "Act 2B", "text": "The hero hits rock bottom with a whiff of death."},
        {"id": "doc2", "concept_name": "Pope in the Pool", "chapter": "Exposition", "text": "Deliver dull exposition during a visually arresting event."},
        {"id": "doc3", "concept_name": "Inciting Incident", "chapter": "Act 1", "text": "The balance of forces is radically upset by an external event."}
    ]
    bm25.index_documents(docs)

    results = bm25.search("Pope in the Pool exposition")
    assert len(results) > 0
    assert results[0][0] == "doc2"

    results_lost = bm25.search("rock bottom whiff of death")
    assert len(results_lost) > 0
    assert results_lost[0][0] == "doc1"


def test_retrieval_mckee_inciting_incident(retriever):
    """Verifies retrieval of Robert McKee's Inciting Incident principle."""
    query = TheorySearchQuery(
        query="How does the inciting incident radically upset the balance of forces according to Robert McKee?",
        author_filter="Robert McKee",
        top_k=3
    )
    response = retriever.search(query)
    assert response.total_matches > 0
    top_chunk = response.results[0].chunk
    assert top_chunk.author == "Robert McKee"
    assert "Inciting Incident" in top_chunk.concept_name
    assert "balance of forces" in top_chunk.content

    # Verify citation
    citation = response.results[0].citation
    assert citation.author == "Robert McKee"
    assert citation.page_number == 181
    assert "Educational Fair Use" in citation.license_status


def test_retrieval_snyder_save_the_cat_beats(retriever):
    """Verifies retrieval of Blake Snyder's Save the Cat beat sheet principles."""
    query = TheorySearchQuery(
        query="What happens at the Midpoint and All Is Lost beat with the whiff of death?",
        top_k=3
    )
    response = retriever.search(query)
    assert response.total_matches > 0
    matched_chunks = [r.chunk for r in response.results]
    assert any("Snyder" in c.author for c in matched_chunks)
    assert any("All Is Lost" in c.content or "Midpoint" in c.content for c in matched_chunks)


def test_retrieval_truby_psychological_vs_moral_need(retriever):
    """Verifies retrieval of John Truby's Psychological vs Moral Need concept."""
    query = TheorySearchQuery(
        query="What is the difference between psychological need and moral need in John Truby?",
        author_filter="John Truby",
        top_k=2
    )
    response = retriever.search(query)
    assert response.total_matches > 0
    top_chunk = response.results[0].chunk
    assert top_chunk.author == "John Truby"
    assert "Psychological Need" in top_chunk.concept_name
    assert "Moral Need" in top_chunk.content


def test_retrieval_field_three_act_paradigm(retriever):
    """Verifies retrieval of Syd Field's Three-Act Paradigm and Plot Points."""
    query = TheorySearchQuery(
        query="How does Syd Field define Plot Point I and Plot Point II in Screenplay?",
        author_filter="Syd Field",
        top_k=3
    )
    response = retriever.search(query)
    assert response.total_matches > 0
    assert any("Plot Point" in r.chunk.concept_name or "Paradigm" in r.chunk.concept_name for r in response.results)


def test_retrieval_vogler_heros_journey_archetypes(retriever):
    """Verifies retrieval of Christopher Vogler's Hero's Journey and Archetypes."""
    query = TheorySearchQuery(
        query="What are the 12 stages of the Hero's Journey and archetypes like the Shadow and Threshold Guardian?",
        author_filter="Christopher Vogler",
        top_k=3
    )
    response = retriever.search(query)
    assert response.total_matches > 0
    matched = [r.chunk for r in response.results]
    assert any("Vogler" in c.author for c in matched)
    assert any("12 Stages" in c.concept_name or "Archetypes" in c.concept_name for c in matched)


def test_retrieval_yorke_into_the_woods(retriever):
    """Verifies retrieval of John Yorke's Fractal Five-Act and Midpoint Mirror."""
    query = TheorySearchQuery(
        query="Why is the midpoint a mirror where the protagonist sees their true self in Into the Woods?",
        author_filter="John Yorke",
        top_k=2
    )
    response = retriever.search(query)
    assert response.total_matches > 0
    top_chunk = response.results[0].chunk
    assert top_chunk.author == "John Yorke"
    assert "Midpoint" in top_chunk.concept_name or "Mirror" in top_chunk.concept_name


def test_retrieval_classical_aristotle_and_freytag(retriever):
    """Verifies retrieval of Aristotelian Peripeteia/Anagnorisis and Freytag's Pyramid."""
    query = TheorySearchQuery(
        query="Aristotle Peripeteia Anagnorisis Catharsis Poetics",
        top_k=3
    )
    response = retriever.search(query)
    assert response.total_matches > 0
    contents = [r.chunk.content for r in response.results]
    assert any("Peripeteia" in c or "Catharsis" in c or "Aristotle" in c for c in contents)


def test_explicit_disclosed_limitations_for_missing_framework(retriever):
    """
    CRITICAL RULE TEST: If a user queries an author or framework not present,
    the system MUST disclose that limitation rather than inventing content.
    """
    query = TheorySearchQuery(
        query="How does Lajos Egri formulate the premise in The Art of Dramatic Writing?",
        author_filter="Lajos Egri",
        top_k=3
    )
    response = retriever.search(query)
    assert response.disclosed_limitations is not None
    assert "Lajos Egri" in response.disclosed_limitations
    assert "Available authors" in response.disclosed_limitations


def test_citation_formatting():
    """Verifies that citations format with full audit provenance."""
    chunk = TheoryChunk(
        chunk_id="test_chunk_1",
        source_id="test_source",
        source_title="The Screenwriter's Bible",
        author="David Trottier",
        chapter_or_section="Book 1: How to Write a Screenplay",
        page_number=45,
        concept_name="Formatting Rules",
        topic_category="scene_design",
        content="Standard screenplay font is 12-point Courier with 1.5-inch left margin.",
        key_takeaways=["Use Courier 12pt", "1.5 inch margin"],
        license_status="Educational Synthesis",
        publication_info="Silman-James Press, 2019"
    )
    citation = CitationEngine.create_citation(chunk, relevance_score=0.95)
    formatted = CitationEngine.format_citation_string(citation)
    assert "[David Trottier, 'The Screenwriter's Bible'" in formatted
    assert "p. 45" in formatted
    assert "Standard screenplay font is 12-point Courier" in formatted


def test_fastapi_search_endpoint(retriever):
    """Verifies search_theory endpoint logic."""
    payload = TheorySearchQuery(
        query="How does dramatic irony create tension in screenplay pacing?",
        top_k=3,
        min_score_threshold=0.2
    )
    response = search_theory(payload, retriever=retriever)
    assert response.total_matches > 0
    assert len(response.results) > 0
    assert response.results[0].citation is not None
    assert response.results[0].citation.source_title is not None


def test_fastapi_sources_endpoint(retriever):
    """Verifies list_theory_sources endpoint logic."""
    sources = list_theory_sources(retriever=retriever)
    assert len(sources) >= 7
    source_titles = [s.title for s in sources]
    assert any("Story" in t for t in source_titles)
    assert any("Save the Cat" in t for t in source_titles)
    assert any("Anatomy of Story" in t for t in source_titles)
    assert any("Screenplay" in t for t in source_titles)
    assert any("Writer's Journey" in t for t in source_titles)
    assert any("Into the Woods" in t for t in source_titles)
    assert any("Classical Dramaturgy" in t for t in source_titles)
