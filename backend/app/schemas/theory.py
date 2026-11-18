"""
Theory Knowledge Base Schemas & DTOs
Strict metadata, citation, and retrieval contracts for Story Analyzer AI.
"""

from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field


class TheorySourceMetadata(BaseModel):
    """Metadata describing a canonical screenwriting source or user reference document."""
    source_id: str = Field(..., description="Unique slug or UUID for the source work")
    title: str = Field(..., description="Full title of the book, paper, or document")
    author: str = Field(..., description="Author(s) or creator(s)")
    publication_year: Optional[int] = Field(None, description="Year of publication")
    publisher: Optional[str] = Field(None, description="Publisher or educational institution")
    edition: Optional[str] = Field(None, description="Edition or volume")
    isbn: Optional[str] = Field(None, description="ISBN if applicable")
    source_url: Optional[str] = Field(None, description="Official publisher or educational link")
    license_status: str = Field(
        ...,
        description="Copyright and usage status (e.g. 'Educational Fair Use / Authorized Synthesis', 'Public Domain', 'User Uploaded')"
    )
    framework_tags: List[str] = Field(
        default_factory=list,
        description="Standard dramaturgical frameworks covered (e.g. 'Three-Act', 'Hero's Journey', '22 Steps', 'Save the Cat')"
    )


class TheoryChunk(BaseModel):
    """An individual indexed knowledge chunk with full citation provenance."""
    chunk_id: str = Field(..., description="Deterministic unique identifier for chunk")
    source_id: str = Field(..., description="References source work")
    source_title: str = Field(..., description="Title of the source work")
    author: str = Field(..., description="Author name")
    chapter_or_section: str = Field(..., description="Chapter, section heading, or module")
    page_number: Optional[int] = Field(None, description="Page number if available from source pagination")
    concept_name: str = Field(..., description="Specific dramaturgical concept (e.g. 'Inciting Incident', 'All Is Lost')")
    topic_category: str = Field(
        ...,
        description="Category: 'structure', 'character', 'conflict', 'scene_design', 'dialogue', 'theme', 'pacing', 'antagonist'"
    )
    content: str = Field(..., description="Verifiable canonical excerpt, authorized summary, or analytical principle")
    key_takeaways: List[str] = Field(default_factory=list, description="Structured bullet points of the rule")
    license_status: str = Field(..., description="Copyright/license status")
    publication_info: str = Field(..., description="Publication details string")
    source_url: Optional[str] = None


class TheoryCitation(BaseModel):
    """Auditable citation linking an analytical insight directly to theory evidence."""
    source_title: str
    author: str
    chapter_or_section: str
    page_number: Optional[int] = None
    concept_name: str
    exact_quote_or_summary: str
    publication_info: str
    source_url: Optional[str] = None
    license_status: str
    relevance_score: float = Field(..., ge=0.0, le=1.0)


class TheorySearchQuery(BaseModel):
    """Query payload for hybrid theory retrieval."""
    query: str = Field(..., description="Dramaturgical query or analytical question")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results to retrieve")
    author_filter: Optional[str] = Field(None, description="Filter by author (e.g. 'Robert McKee')")
    concept_filter: Optional[str] = Field(None, description="Filter by concept (e.g. 'Midpoint')")
    category_filter: Optional[str] = Field(None, description="Filter by category (e.g. 'structure')")
    framework_filter: Optional[str] = Field(None, description="Filter by framework tag (e.g. 'Save the Cat')")
    min_score_threshold: float = Field(default=0.35, ge=0.0, le=1.0)


class TheorySearchResult(BaseModel):
    """Individual retrieved chunk with hybrid scoring metrics."""
    chunk: TheoryChunk
    dense_score: float
    sparse_score: float
    combined_score: float
    citation: TheoryCitation


class TheorySearchResponse(BaseModel):
    """API response contract for knowledge retrieval."""
    query: str
    total_matches: int
    results: List[TheorySearchResult]
    available_frameworks_consulted: List[str]
    disclosed_limitations: Optional[str] = Field(
        None,
        description="Explicit notice if a requested framework/author is not available in the knowledge base"
    )


class IngestDocumentRequest(BaseModel):
    """Request metadata for user-uploaded reference documents."""
    title: str
    author: str
    publication_info: Optional[str] = "User-Provided Reference Document"
    license_status: str = "User Authorized Reference Material"
    framework_tags: List[str] = Field(default_factory=list)

