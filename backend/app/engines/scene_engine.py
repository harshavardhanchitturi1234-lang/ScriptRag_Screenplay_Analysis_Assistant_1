"""
Scene Extraction & Structural Boundary Engine.
Builds linked SceneData objects from ScreenplayElements with 1/8th-page calculations,
dialogue block grouping, character presence detection, and bidirectional linking.
"""

import uuid
import re
from typing import List, Dict, Any, Optional
from app.schemas.screenplay import ScreenplayElement, ElementType, SceneData, DialogueBlock
from app.parsers.slugline_parser import SluglineParser


class SceneEngine:
    """Engine responsible for segmenting, structuring, and linking screenplay scenes."""

    LINES_PER_EIGHTH = 7.0  # Industry rule of thumb: ~55 lines/page / 8 = ~7 lines per 1/8th

    @classmethod
    def extract_scenes(cls, elements: List[ScreenplayElement]) -> List[SceneData]:
        """
        Groups screenplay elements into structured SceneData objects.
        Links previous_scene_id and next_scene_id.
        """
        scenes: List[SceneData] = []
        current_scene_elements: List[ScreenplayElement] = []
        current_heading_element: Optional[ScreenplayElement] = None
        scene_counter = 0

        for el in elements:
            if el.element_type == ElementType.SCENE_HEADING:
                if current_heading_element is not None:
                    scene_counter += 1
                    scene = cls._build_scene(
                        heading_el=current_heading_element,
                        body_elements=current_scene_elements,
                        scene_seq=scene_counter
                    )
                    scenes.append(scene)
                    current_scene_elements = []

                current_heading_element = el
            else:
                if current_heading_element is not None:
                    current_scene_elements.append(el)

        # Process the final scene
        if current_heading_element is not None:
            scene_counter += 1
            scene = cls._build_scene(
                heading_el=current_heading_element,
                body_elements=current_scene_elements,
                scene_seq=scene_counter
            )
            scenes.append(scene)

        # Establish bidirectional links (previous_scene_id and next_scene_id)
        for i in range(len(scenes)):
            if i > 0:
                scenes[i].previous_scene_id = scenes[i - 1].scene_id
            if i < len(scenes) - 1:
                scenes[i].next_scene_id = scenes[i + 1].scene_id

        return scenes

    @classmethod
    def _build_scene(
        cls,
        heading_el: ScreenplayElement,
        body_elements: List[ScreenplayElement],
        scene_seq: int
    ) -> SceneData:
        """Constructs a single SceneData instance."""
        slug_info = SluglineParser.parse(heading_el.text, fallback_scene_number=scene_seq)

        # Page boundaries
        page_start = float(heading_el.page_number)
        page_end = float(body_elements[-1].page_number) if body_elements else page_start

        # Calculate line count and 1/8th page duration
        total_lines = len(body_elements) + 1
        eighths = max(1, round(total_lines / cls.LINES_PER_EIGHTH))

        # Extract Dialogue blocks, Action descriptions, and Characters
        dialogue_blocks: List[DialogueBlock] = []
        action_paragraphs: List[str] = []
        speaking_characters: set[str] = set()

        i = 0
        n = len(body_elements)
        while i < n:
            el = body_elements[i]
            if el.element_type == ElementType.ACTION:
                action_paragraphs.append(el.text)
                i += 1
            elif el.element_type == ElementType.CHARACTER:
                raw_char = el.text.strip()
                # Extract clean character name and extension
                ext_match = re.search(r"\s*(\([^\)]+\))$", raw_char)
                extension = ext_match.group(1) if ext_match else None
                clean_name = re.sub(r"\s*\([^\)]+\)$", "", raw_char).strip().upper()
                speaking_characters.add(clean_name)

                # Lookahead for parenthetical & dialogue
                parenthetical = None
                dialogue_lines = []
                line_start = el.line_number
                line_end = el.line_number
                i += 1

                while i < n and body_elements[i].element_type in [ElementType.PARENTHETICAL, ElementType.DIALOGUE]:
                    sub_el = body_elements[i]
                    line_end = sub_el.line_number
                    if sub_el.element_type == ElementType.PARENTHETICAL:
                        parenthetical = sub_el.text
                    else:
                        dialogue_lines.append(sub_el.text)
                    i += 1

                if dialogue_lines:
                    dialogue_blocks.append(DialogueBlock(
                        character=clean_name,
                        character_extension=extension,
                        parenthetical=parenthetical,
                        text=" ".join(dialogue_lines),
                        page_number=el.page_number,
                        line_start=line_start,
                        line_end=line_end
                    ))
            else:
                i += 1

        action_text = "\n\n".join(action_paragraphs)

        # Detect non-speaking characters explicitly mentioned in action text (ALL CAPS names)
        non_speaking_chars = cls._extract_action_characters(action_text, speaking_characters)

        all_characters_present = sorted(list(speaking_characters.union(non_speaking_chars)))

        # Extract salient narrative events from action lines
        important_events = [p for p in action_paragraphs if len(p.split()) > 6][:4]

        scene_id = str(uuid.uuid4())

        return SceneData(
            scene_id=scene_id,
            scene_number=slug_info["scene_number"],
            slugline=slug_info["slugline"],
            int_ext=slug_info["int_ext"],
            location=slug_info["location"],
            time_of_day=slug_info["time_of_day"],
            page_start=page_start,
            page_end=page_end,
            eighths_length=eighths,
            characters_present=all_characters_present,
            speaking_characters=sorted(list(speaking_characters)),
            non_speaking_characters=sorted(list(non_speaking_chars)),
            action_description=action_text,
            dialogue=dialogue_blocks,
            important_events=important_events,
            previous_scene_id=None,
            next_scene_id=None
        )

    @classmethod
    def _extract_action_characters(cls, action_text: str, speaking_characters: set[str]) -> set[str]:
        """Finds non-speaking characters introduced in all-caps within scene descriptions."""
        non_speaking = set()
        # Find uppercase words of 3+ letters that look like names
        words = re.findall(r"\b[A-Z]{3,20}\b", action_text)
        ignore_words = {
            "THE", "AND", "WITH", "FROM", "INTO", "THROUGH", "ACROSS", "CAMERA",
            "ANGLE", "CLOSE", "WIDE", "VIEW", "SFX", "VFX", "POV", "INSERT",
            "BEAT", "MOMENTS", "LATER", "SUDDENLY", "LOOKS", "WALKS", "STANDS"
        }
        for w in words:
            if w not in ignore_words and w not in speaking_characters:
                # Add if appears as a capitalized actor or distinct noun
                if w.isalpha():
                    non_speaking.add(w)
        return non_speaking
