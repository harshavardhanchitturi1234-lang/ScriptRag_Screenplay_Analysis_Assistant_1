"""
Document Extractors for Screenwriting RAG Pipeline
Supports PDF, DOCX, TXT, and Markdown documents with page and section tracking.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import pypdf
import docx

from app.rag.text_cleaner import TextCleaner


class ExtractedPage(BaseModel):
    page_number: int
    text: str
    word_count: int


class DocumentExtractionResult(BaseModel):
    filename: str
    file_type: str
    total_pages: int
    total_words: int
    pages: List[ExtractedPage]
    raw_text: str
    metadata: Dict[str, Any]


class DocumentExtractor:
    """Unified multi-format document extraction service."""

    @classmethod
    def extract_file(cls, file_path: str | Path) -> DocumentExtractionResult:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Document file not found: {file_path}")

        ext = path.suffix.lower()
        if ext == ".pdf":
            return cls.extract_pdf(path)
        elif ext in [".docx", ".doc"]:
            return cls.extract_docx(path)
        elif ext in [".txt", ".fountain", ".md"]:
            return cls.extract_text(path)
        else:
            raise ValueError(f"Unsupported document format '{ext}'. Supported: .pdf, .docx, .txt, .fountain, .md")

    @classmethod
    def extract_pdf(cls, file_path: Path) -> DocumentExtractionResult:
        """Extracts text page-by-page from PDF with pypdf."""
        pages: List[ExtractedPage] = []
        raw_text_parts = []
        metadata: Dict[str, Any] = {}

        with open(file_path, "rb") as f:
            reader = pypdf.PdfReader(f)
            if reader.metadata:
                metadata = {
                    "title": reader.metadata.title or file_path.stem,
                    "author": reader.metadata.author or "Unknown",
                    "creator": reader.metadata.creator or "",
                    "producer": reader.metadata.producer or ""
                }

            total_pages = len(reader.pages)
            for idx, page in enumerate(reader.pages, start=1):
                raw_page_text = page.extract_text() or ""
                cleaned_page_text = TextCleaner.clean(raw_page_text, preserve_page_breaks=False)
                words = len(cleaned_page_text.split())
                pages.append(ExtractedPage(page_number=idx, text=cleaned_page_text, word_count=words))
                raw_text_parts.append(cleaned_page_text)

        full_raw_text = "\n\n".join(raw_text_parts)
        total_words = sum(p.word_count for p in pages)

        return DocumentExtractionResult(
            filename=file_path.name,
            file_type="PDF",
            total_pages=total_pages,
            total_words=total_words,
            pages=pages,
            raw_text=full_raw_text,
            metadata=metadata
        )

    @classmethod
    def extract_docx(cls, file_path: Path) -> DocumentExtractionResult:
        """Extracts text and headings from DOCX using python-docx."""
        doc = docx.Document(str(file_path))
        metadata: Dict[str, Any] = {
            "title": doc.core_properties.title or file_path.stem,
            "author": doc.core_properties.author or "Unknown",
            "category": doc.core_properties.category or ""
        }

        paragraphs_text = []
        for p in doc.paragraphs:
            if p.text.strip():
                paragraphs_text.append(p.text.strip())

        # Also extract table text
        for table in doc.tables:
            for row in table.rows:
                row_text = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if row_text:
                    paragraphs_text.append(" | ".join(row_text))

        full_raw_text = TextCleaner.clean("\n\n".join(paragraphs_text), preserve_page_breaks=False)
        words = len(full_raw_text.split())

        # Approximate pages for DOCX (standard 250 words per page estimation)
        est_pages = max(1, (words + 249) // 250)
        pages = [
            ExtractedPage(page_number=1, text=full_raw_text, word_count=words)
        ]

        return DocumentExtractionResult(
            filename=file_path.name,
            file_type="DOCX",
            total_pages=est_pages,
            total_words=words,
            pages=pages,
            raw_text=full_raw_text,
            metadata=metadata
        )

    @classmethod
    def extract_text(cls, file_path: Path) -> DocumentExtractionResult:
        """Extracts plaintext or Markdown file."""
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            raw_content = f.read()

        cleaned = TextCleaner.clean(raw_content, preserve_page_breaks=True)
        # Check if text contains explicit [PAGE_BREAK] markers
        if "[PAGE_BREAK]" in cleaned:
            page_chunks = cleaned.split("[PAGE_BREAK]")
            pages = []
            for idx, p_text in enumerate(page_chunks, start=1):
                clean_p = p_text.strip()
                pages.append(ExtractedPage(page_number=idx, text=clean_p, word_count=len(clean_p.split())))
        else:
            words = len(cleaned.split())
            pages = [ExtractedPage(page_number=1, text=cleaned, word_count=words)]

        return DocumentExtractionResult(
            filename=file_path.name,
            file_type="TEXT",
            total_pages=len(pages),
            total_words=sum(p.word_count for p in pages),
            pages=pages,
            raw_text=cleaned.replace("[PAGE_BREAK]", "\n"),
            metadata={"title": file_path.stem}
        )

