"""
Unified Screenplay Parser & AST Builder.
Parses normalized lines into ScreenplayElements, preserving line numbers, page numbers,
and formatting indentation coordinates.
"""

import re
from typing import List
from app.schemas.screenplay import ScreenplayElement, ElementType
from app.parsers.slugline_parser import SluglineParser


class ScreenplayParser:
    """Core screenplay layout analyzer supporting industry standard conventions."""

    CHARACTER_CUE_REGEX = re.compile(
        r"^[A-Z0-9\s'–—\.\#\-]+(?:\s*\([A-Za-z0-9\s\.\'\/\-]+\))?$"
    )
    TRANSITION_REGEX = re.compile(
        r"^(?:(?:SMASH|DISSOLVE|MATCH|JUMP|QUICK)?\s*CUT\s+TO:|FADE\s+IN:|FADE\s+OUT\.|BLACKOUT\.|THE\s+END)$",
        re.IGNORECASE
    )
    PARENTHETICAL_REGEX = re.compile(r"^\(.*\)$")

    @classmethod
    def parse_document_text(cls, text: str, initial_page: int = 1) -> List[ScreenplayElement]:
        """
        Parses full script text into a sequence of ScreenplayElements.
        Handles page numbers via explicit [PAGE_BREAK] markers or line density counting.
        """
        elements: List[ScreenplayElement] = []
        lines = text.split("\n")

        current_page = initial_page
        line_num = 0

        i = 0
        total_lines = len(lines)

        while i < total_lines:
            raw_line = lines[i]
            line_num += 1
            stripped = raw_line.strip()

            # 1. Page breaks
            if "[PAGE_BREAK]" in stripped or stripped.startswith("==="):
                current_page += 1
                elements.append(ScreenplayElement(
                    element_type=ElementType.PAGE_BREAK,
                    text="PAGE_BREAK",
                    page_number=current_page,
                    line_number=line_num,
                    raw_text=raw_line
                ))
                i += 1
                continue

            if not stripped:
                i += 1
                continue

            # 2. Scene Headings (Sluglines)
            if SluglineParser.is_slugline(stripped):
                elements.append(ScreenplayElement(
                    element_type=ElementType.SCENE_HEADING,
                    text=stripped,
                    page_number=current_page,
                    line_number=line_num,
                    raw_text=raw_line
                ))
                i += 1
                continue

            # 3. Transitions (e.g. CUT TO:, FADE OUT.)
            if cls.TRANSITION_REGEX.match(stripped) and len(stripped) < 35:
                elements.append(ScreenplayElement(
                    element_type=ElementType.TRANSITION,
                    text=stripped,
                    page_number=current_page,
                    line_number=line_num,
                    raw_text=raw_line
                ))
                i += 1
                continue

            # 4. Character Cue followed by dialogue
            # Screenplay Character names are uppercase, length 2 to 38, not ending in periods/commas
            is_char_candidate = (
                cls._is_character_line(stripped)
                and (i + 1 < total_lines and lines[i + 1].strip())
            )

            if is_char_candidate:
                char_text = stripped
                elements.append(ScreenplayElement(
                    element_type=ElementType.CHARACTER,
                    text=char_text,
                    page_number=current_page,
                    line_number=line_num,
                    raw_text=raw_line
                ))
                i += 1

                # Parse subsequent parentheticals & dialogue block lines
                while i < total_lines and lines[i].strip():
                    line_num += 1
                    inner_raw = lines[i]
                    inner_strip = inner_raw.strip()

                    # Stop if next line is another scene heading
                    if SluglineParser.is_slugline(inner_strip):
                        break

                    if cls.PARENTHETICAL_REGEX.match(inner_strip):
                        elements.append(ScreenplayElement(
                            element_type=ElementType.PARENTHETICAL,
                            text=inner_strip,
                            page_number=current_page,
                            line_number=line_num,
                            raw_text=inner_raw
                        ))
                    else:
                        elements.append(ScreenplayElement(
                            element_type=ElementType.DIALOGUE,
                            text=inner_strip,
                            page_number=current_page,
                            line_number=line_num,
                            raw_text=inner_raw
                        ))
                    i += 1
                continue

            # 5. Action / Scene Description
            elements.append(ScreenplayElement(
                element_type=ElementType.ACTION,
                text=stripped,
                page_number=current_page,
                line_number=line_num,
                raw_text=raw_line
            ))
            i += 1

        return elements

    @classmethod
    def _is_character_line(cls, line: str) -> bool:
        """Heuristic check for speaking character name lines."""
        if not line or len(line) > 40:
            return False

        # Must not end in standard sentence punctuation
        if line.endswith((".", ",", "!", "?", "--", ";")):
            return False

        # Extract name without extension (V.O.), (O.S.), (CONT'D)
        clean_name = re.sub(r"\s*\([^\)]+\)", "", line).strip()
        if not clean_name:
            return False

        # Must be uppercase letters
        letters = [c for c in clean_name if c.isalpha()]
        if not letters:
            return False

        # Character names should be >80% uppercase
        upper_count = sum(1 for c in letters if c.isupper())
        if (upper_count / len(letters)) < 0.85:
            return False

        # Common action verbs that appear at the start of lines to avoid false positives
        action_verb_starters = {"SUDDENLY", "THEN", "MEANWHILE", "LATER", "INSIDE", "OUTSIDE"}
        if clean_name.upper() in action_verb_starters:
            return False

        return bool(cls.CHARACTER_CUE_REGEX.match(line))
