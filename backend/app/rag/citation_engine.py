"""
Theory Citation Engine for Story Analyzer AI.
Generates auditable, citation-grounded references adhering to copyright & educational fair-use rules.
"""

from typing import Optional, Dict, Any
from app.schemas.theory import TheoryCitation, TheoryChunk


class CitationEngine:
    """Creates verifiable citations from retrieved theory chunks."""

    @classmethod
    def create_citation(
        cls,
        chunk: TheoryChunk,
        relevance_score: float,
        specific_quote: Optional[str] = None
    ) -> TheoryCitation:
        """
        Builds an auditable citation object.
        If a specific quote is provided, it uses it; otherwise it takes the most representative sentence or permitted summary.
        """
        if specific_quote:
            quote_text = specific_quote.strip()
        else:
            # Select the first 1-2 salient sentences from chunk content
            sentences = chunk.content.split(". ")
            quote_text = ". ".join(sentences[:2]).strip()
            if not quote_text.endswith("."):
                quote_text += "."

        return TheoryCitation(
            source_title=chunk.source_title,
            author=chunk.author,
            chapter_or_section=chunk.chapter_or_section,
            page_number=chunk.page_number,
            concept_name=chunk.concept_name,
            exact_quote_or_summary=quote_text,
            publication_info=chunk.publication_info,
            source_url=chunk.source_url,
            license_status=chunk.license_status,
            relevance_score=round(relevance_score, 4)
        )

    @classmethod
    def format_citation_string(cls, citation: TheoryCitation) -> str:
        """Formats citation for human-readable report or screenplay viewer."""
        base = f"[{citation.author}, '{citation.source_title}'"
        if citation.chapter_or_section:
            base += f", {citation.chapter_or_section}"
        if citation.page_number:
            base += f", p. {citation.page_number}"
        base += f"]: \"{citation.exact_quote_or_summary}\""
        return base

