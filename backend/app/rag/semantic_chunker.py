"""
Semantic Chunker for Screenwriting Knowledge Base
Chunks documents along structural headings, concepts, and paragraph boundaries.
Attaches strict citation metadata to every chunk.
"""

import re
import uuid
import hashlib
from typing import List, Dict, Any, Optional
from app.schemas.theory import TheoryChunk, TheorySourceMetadata


class SemanticChunker:
    """
    Intelligent structural chunker for screenwriting literature and educational materials.
    Preserves heading context, concept names, and exact page references.
    """

    HEADING_PATTERNS = [
        re.compile(r"^(?:#{1,4}\s+|[A-Z\s]{4,}:|\b(?:CHAPTER|ACT|SEQUENCE|STEP|PART|BEAT)\s+[0-9IVXLCDM]+[:.\s-]*)(.*)$", re.MULTILINE),
        re.compile(r"^([A-Z][A-Za-z0-9\s,–—'\"]{3,60})\n(?:={3,}|-{3,})$", re.MULTILINE)
    ]

    CONCEPT_KEYWORDS = {
        "structure": ["three-act", "five-act", "inciting incident", "plot point", "midpoint", "climax", "resolution", "beat sheet", "sequence", "save the cat", "catalyst", "debate", "break into two", "all is lost", "dark night"],
        "character": ["protagonist", "antagonist", "want vs need", "character flaw", "fatal flaw", "character arc", "transformation", "archetype", "shadow", "mentor", "herald", "threshold guardian"],
        "conflict": ["stakes", "dramatic tension", "conflict", "obstacle", "dilemma", "crisis", "antagonistic force", "four-corner opposition"],
        "scene_design": ["scene construction", "slugline", "turning point", "gap", "objective", "beat", "polarity shift", "scene objective"],
        "dialogue": ["subtext", "exposition", "voice", "on-the-nose", "pope in the pool", "text vs subtext"],
        "theme": ["controlling idea", "moral argument", "theme stated", "central dramatic question", "premise"],
        "pacing": ["pacing", "momentum", "ticking clock", "compression", "tempo", "rhythm"]
    }

    def __init__(self, target_word_count: int = 350, max_word_count: int = 550, overlap_words: int = 50):
        self.target_word_count = target_word_count
        self.max_word_count = max_word_count
        self.overlap_words = overlap_words

    def infer_category_and_concept(self, text: str, heading: str) -> tuple[str, str]:
        """Infers the most appropriate dramaturgical category and concept name."""
        combined = f"{heading} {text}".lower()
        
        # Check specific keywords
        best_category = "structure"
        highest_matches = 0

        for cat, keywords in self.CONCEPT_KEYWORDS.items():
            matches = sum(1 for kw in keywords if kw in combined)
            if matches > highest_matches:
                highest_matches = matches
                best_category = cat

        # Determine concept name from heading or key phrase
        if heading and heading != "General":
            clean_heading = re.sub(r"^#+\s*", "", heading).strip()
            concept_name = clean_heading.split(" - ")[0].split(":")[0].strip()
        else:
            # Fallback to first matched concept keyword
            concept_name = "Core Screenwriting Principle"
            for keywords in self.CONCEPT_KEYWORDS.values():
                for kw in keywords:
                    if kw in combined:
                        concept_name = kw.title()
                        break

        return best_category, concept_name

    def chunk_document(
        self,
        text: str,
        source_metadata: TheorySourceMetadata,
        page_mapping: Optional[List[dict]] = None
    ) -> List[TheoryChunk]:
        """
        Chunks text while maintaining structural context and page references.
        """
        chunks: List[TheoryChunk] = []
        
        # Split into structural sections based on markdown/text headings
        sections = self._split_by_headings(text)

        for sec_heading, sec_text, approx_page in sections:
            sec_words = sec_text.split()
            if not sec_words:
                continue

            category, concept = self.infer_category_and_concept(sec_text, sec_heading)

            if len(sec_words) <= self.max_word_count:
                chunks.append(self._create_chunk(
                    content=sec_text.strip(),
                    heading=sec_heading,
                    concept=concept,
                    category=category,
                    page=approx_page,
                    source=source_metadata
                ))
            else:
                # Sub-chunk by paragraphs with overlap
                sub_chunks = self._sliding_window_chunk(sec_text, sec_words)
                for idx, sub_text in enumerate(sub_chunks):
                    sub_concept = f"{concept} (Part {idx + 1})" if len(sub_chunks) > 1 else concept
                    chunks.append(self._create_chunk(
                        content=sub_text.strip(),
                        heading=sec_heading,
                        concept=sub_concept,
                        category=category,
                        page=approx_page,
                        source=source_metadata
                    ))

        return chunks

    def _split_by_headings(self, text: str) -> List[tuple[str, str, Optional[int]]]:
        """Splits full document into structural sections by heading or page breaks."""
        lines = text.split("\n")
        sections = []
        current_heading = "Overview"
        current_lines = []
        current_page = 1

        for line in lines:
            if "[PAGE_BREAK]" in line:
                current_page += 1
                continue

            # Check if line is a header
            header_match = False
            for pat in self.HEADING_PATTERNS:
                m = pat.match(line.strip())
                if m:
                    if current_lines:
                        sections.append((current_heading, "\n".join(current_lines), current_page))
                        current_lines = []
                    current_heading = line.strip()
                    header_match = True
                    break

            if not header_match:
                current_lines.append(line)

        if current_lines:
            sections.append((current_heading, "\n".join(current_lines), current_page))

        return sections

    def _sliding_window_chunk(self, text: str, words: list[str]) -> List[str]:
        """Sliding window paragraph-aware chunker."""
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks = []
        curr_paras = []
        curr_words = 0

        for p in paragraphs:
            p_words = len(p.split())
            if curr_words + p_words > self.max_word_count and curr_paras:
                chunks.append("\n\n".join(curr_paras))
                # Keep last paragraph for overlap if reasonable
                if p_words < self.target_word_count:
                    curr_paras = [curr_paras[-1], p] if len(curr_paras) > 1 else [p]
                    curr_words = sum(len(x.split()) for x in curr_paras)
                else:
                    curr_paras = [p]
                    curr_words = p_words
            else:
                curr_paras.append(p)
                curr_words += p_words

        if curr_paras:
            chunks.append("\n\n".join(curr_paras))

        return chunks

    def _create_chunk(
        self,
        content: str,
        heading: str,
        concept: str,
        category: str,
        page: Optional[int],
        source: TheorySourceMetadata
    ) -> TheoryChunk:
        """Constructs a deterministic TheoryChunk object."""
        # Generate deterministic UUID based on source + heading + content prefix
        hash_input = f"{source.source_id}:{heading}:{content[:100]}"
        chunk_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, hash_input))

        # Extract 2-3 key takeaways from content
        sentences = [s.strip() for s in re.split(r"[.!?]\s+", content) if len(s.strip()) > 20]
        takeaways = sentences[:3] if sentences else [f"Analytical principle regarding {concept}"]

        pub_info = f"{source.author}, '{source.title}'"
        if source.publisher:
            pub_info += f" ({source.publisher}"
            if source.publication_year:
                pub_info += f", {source.publication_year}"
            pub_info += ")"

        return TheoryChunk(
            chunk_id=chunk_uuid,
            source_id=source.source_id,
            source_title=source.title,
            author=source.author,
            chapter_or_section=heading,
            page_number=page,
            concept_name=concept,
            topic_category=category,
            content=content,
            key_takeaways=takeaways,
            license_status=source.license_status,
            publication_info=pub_info,
            source_url=source.source_url
        )

