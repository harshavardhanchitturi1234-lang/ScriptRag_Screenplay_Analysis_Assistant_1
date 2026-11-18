"""
Text cleaning and normalization utility for Screenwriting Knowledge Base & Scripts.
Handles typographic ligatures, smart quotes, page headers/footers, and whitespace.
"""

import re
import unicodedata


class TextCleaner:
    """Standardized text normalization for RAG ingestion."""

    # Typographic replacements for clean search and embedding fidelity
    CHAR_REPLACEMENTS = {
        "\u2018": "'",  # Left single quote
        "\u2019": "'",  # Right single quote
        "\u201c": '"',  # Left double quote
        "\u201d": '"',  # Right double quote
        "\u2014": " -- ",  # Em dash
        "\u2013": " - ",   # En dash
        "\u2026": "...",  # Ellipsis
        "\u00a0": " ",    # Non-breaking space
        "\ufeff": "",     # Byte Order Mark (BOM)
        "\x0c": "\n[PAGE_BREAK]\n",  # Form feed / PDF page boundary
    }

    @classmethod
    def clean(cls, text: str, preserve_page_breaks: bool = True) -> str:
        """
        Cleans and normalizes text for ingestion.
        
        Args:
            text: Raw input string.
            preserve_page_breaks: If True, preserves [PAGE_BREAK] tokens for page mapping.
            
        Returns:
            Cleaned and normalized text string.
        """
        if not text:
            return ""

        # Normalize unicode (NFKC)
        cleaned = unicodedata.normalize("NFKC", text)

        # Apply character mappings
        for orig, replacement in cls.CHAR_REPLACEMENTS.items():
            if orig == "\x0c" and not preserve_page_breaks:
                cleaned = cleaned.replace(orig, "\n")
            else:
                cleaned = cleaned.replace(orig, replacement)

        # Normalize line endings to \n
        cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")

        # Collapse excess empty lines (max 2 consecutive newlines)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

        # Remove trailing whitespace per line
        cleaned = "\n".join(line.rstrip() for line in cleaned.split("\n"))

        return cleaned.strip()

    @classmethod
    def remove_boilerplate_headers(cls, lines: list[str]) -> list[str]:
        """
        Removes repeated running page headers and footers from documents.
        """
        cleaned_lines = []
        for line in lines:
            stripped = line.strip()
            # Skip lone page number lines (e.g. "Page 12", "- 12 -", "12.")
            if re.match(r"^(page\s+\d+|\d+\.?|-\s*\d+\s*-)$", stripped, re.IGNORECASE):
                continue
            cleaned_lines.append(line)
        return cleaned_lines

