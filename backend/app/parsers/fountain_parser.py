"""
Fountain Screenplay Format Parser.
Implements the official Fountain standard syntax specification:
http://fountain.io/syntax
"""

import re
from typing import List
from app.schemas.screenplay import ScreenplayElement, ElementType
from app.parsers.slugline_parser import SluglineParser


class FountainParser:
    """Parser for .fountain screenplay format."""

    TRANSITION_REGEX = re.compile(r"^(?:[A-Z\s]+TO:|>.*|<.*>)$")
    PARENTHETICAL_REGEX = re.compile(r"^\(.*\)$")

    @classmethod
    def parse_text(cls, text: str) -> List[ScreenplayElement]:
        """Parses a Fountain script into a stream of ScreenplayElements."""
        elements: List[ScreenplayElement] = []
        lines = text.split("\n")

        current_page = 1
        lines_on_page = 0
        MAX_LINES_PER_PAGE = 55  # Standard industry screenplay page length

        i = 0
        num_lines = len(lines)

        while i < num_lines:
            raw_line = lines[i]
            stripped = raw_line.strip()
            lines_on_page += 1

            # Estimate page increment
            if lines_on_page >= MAX_LINES_PER_PAGE:
                current_page += 1
                lines_on_page = 0

            # 1. Page breaks
            if stripped.startswith("==="):
                current_page += 1
                lines_on_page = 0
                elements.append(ScreenplayElement(
                    element_type=ElementType.PAGE_BREAK,
                    text="PAGE_BREAK",
                    page_number=current_page,
                    line_number=i + 1,
                    raw_text=raw_line
                ))
                i += 1
                continue

            # 2. Skip empty lines, notes ([[ ... ]]), or boneyard (/* ... */)
            if not stripped or stripped.startswith("[[") or stripped.startswith("/*"):
                i += 1
                continue

            # 3. Forced or standard Scene Headings
            if stripped.startswith(".") and not stripped.startswith(".."):
                # Forced scene heading: e.g. '.OPENING SEQUENCE'
                elements.append(ScreenplayElement(
                    element_type=ElementType.SCENE_HEADING,
                    text=stripped[1:].strip(),
                    page_number=current_page,
                    line_number=i + 1,
                    raw_text=raw_line
                ))
                i += 1
                continue
            elif SluglineParser.is_slugline(stripped):
                elements.append(ScreenplayElement(
                    element_type=ElementType.SCENE_HEADING,
                    text=stripped,
                    page_number=current_page,
                    line_number=i + 1,
                    raw_text=raw_line
                ))
                i += 1
                continue

            # 4. Transitions: e.g. 'CUT TO:', 'FADE OUT.', or forced '> FADE IN:'
            if stripped.startswith(">") and not stripped.endswith("<"):
                elements.append(ScreenplayElement(
                    element_type=ElementType.TRANSITION,
                    text=stripped[1:].strip(),
                    page_number=current_page,
                    line_number=i + 1,
                    raw_text=raw_line
                ))
                i += 1
                continue
            elif cls.TRANSITION_REGEX.match(stripped) and len(stripped) < 40:
                elements.append(ScreenplayElement(
                    element_type=ElementType.TRANSITION,
                    text=stripped,
                    page_number=current_page,
                    line_number=i + 1,
                    raw_text=raw_line
                ))
                i += 1
                continue

            # 5. Character Name: all uppercase line (or forced '@') followed immediately by dialogue or parenthetical
            is_forced_char = stripped.startswith("@")
            is_uppercase = stripped.isupper() and len(stripped) < 45 and not stripped.endswith((".", ":", "!", "?", "--"))

            if (is_forced_char or is_uppercase) and (i + 1 < num_lines and lines[i + 1].strip()):
                char_text = stripped[1:].strip() if is_forced_char else stripped
                elements.append(ScreenplayElement(
                    element_type=ElementType.CHARACTER,
                    text=char_text,
                    page_number=current_page,
                    line_number=i + 1,
                    raw_text=raw_line
                ))
                i += 1

                # Parse subsequent parentheticals and dialogue lines
                while i < num_lines and lines[i].strip():
                    inner_line = lines[i]
                    inner_stripped = inner_line.strip()

                    if cls.PARENTHETICAL_REGEX.match(inner_stripped):
                        elements.append(ScreenplayElement(
                            element_type=ElementType.PARENTHETICAL,
                            text=inner_stripped,
                            page_number=current_page,
                            line_number=i + 1,
                            raw_text=inner_line
                        ))
                    else:
                        elements.append(ScreenplayElement(
                            element_type=ElementType.DIALOGUE,
                            text=inner_stripped,
                            page_number=current_page,
                            line_number=i + 1,
                            raw_text=inner_line
                        ))
                    i += 1
                continue

            # 6. Action / Description
            elements.append(ScreenplayElement(
                element_type=ElementType.ACTION,
                text=stripped,
                page_number=current_page,
                line_number=i + 1,
                raw_text=raw_line
            ))
            i += 1

        return elements
