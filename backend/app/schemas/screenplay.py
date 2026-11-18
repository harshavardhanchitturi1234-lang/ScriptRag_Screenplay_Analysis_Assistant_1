"""
Screenplay Abstract Syntax Tree (AST) & DTO Schemas.
Provides strict representations for Screenplay Elements, Scenes, Dialogue, and Characters.
"""

from typing import List, Optional, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field


class ElementType(str, Enum):
    SCENE_HEADING = "SCENE_HEADING"
    ACTION = "ACTION"
    CHARACTER = "CHARACTER"
    PARENTHETICAL = "PARENTHETICAL"
    DIALOGUE = "DIALOGUE"
    TRANSITION = "TRANSITION"
    PAGE_BREAK = "PAGE_BREAK"


class ScreenplayElement(BaseModel):
    """An individual token/element in the screenplay AST."""
    element_type: ElementType
    text: str
    page_number: int
    line_number: int
    raw_text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DialogueBlock(BaseModel):
    """Grouped dialogue unit comprising character, optional parenthetical, and spoken text."""
    character: str
    character_extension: Optional[str] = None  # e.g. (V.O.), (O.S.), (CONT'D)
    parenthetical: Optional[str] = None
    text: str
    page_number: int
    line_start: int
    line_end: int


class SceneData(BaseModel):
    """Complete structured scene unit with bidirectional scene linking."""
    scene_id: str
    scene_number: int
    slugline: str
    int_ext: str  # INT, EXT, INT/EXT
    location: str
    time_of_day: str  # DAY, NIGHT, DUSK, DAWN, CONTINUOUS, etc.
    page_start: float
    page_end: float
    eighths_length: int  # Screenplay page length in 1/8ths
    characters_present: List[str]
    speaking_characters: List[str]
    non_speaking_characters: List[str] = Field(default_factory=list)
    action_description: str
    dialogue: List[DialogueBlock]
    important_events: List[str] = Field(default_factory=list)
    previous_scene_id: Optional[str] = None
    next_scene_id: Optional[str] = None


class CharacterRecord(BaseModel):
    """Character registry entry tracking appearances, dialogue stats, and textual evidence."""
    character_id: str
    canonical_name: str
    aliases: List[str] = Field(default_factory=list)
    first_appearance: Dict[str, Any]  # {"scene_number": 1, "page_number": 1.0, "scene_id": "..."}
    last_appearance: Dict[str, Any]   # {"scene_number": 45, "page_number": 112.0, "scene_id": "..."}
    total_dialogue_blocks: int
    total_words: int
    scenes_present: List[int] = Field(default_factory=list)
    relationships: Dict[str, int] = Field(
        default_factory=dict,
        description="Co-occurrence counts with other characters: {'EVELYN': 18}"
    )
    goals: List[str] = Field(default_factory=list, description="Explicit stated or actively pursued objectives")
    stated_motivations: List[str] = Field(default_factory=list, description="Exact quotes stating reason or motivation")
    actions: List[str] = Field(default_factory=list, description="Key physical actions performed")
    important_decisions: List[str] = Field(default_factory=list, description="Crucial decisions made in scenes")
    character_changes: List[str] = Field(default_factory=list, description="Observed behavioral shifts")


class ScreenplayMetadata(BaseModel):
    """Screenplay summary and ingestion status."""
    screenplay_id: str
    title: str
    author: str = "Unknown"
    format: str  # PDF, DOCX, TXT, FOUNTAIN
    file_size_bytes: int
    file_hash: str
    page_count: int
    scene_count: int
    character_count: int
    status: str  # PENDING, PARSING, PARSED, FAILED
    error_message: Optional[str] = None
    created_at: str


class ScreenplayParsedResponse(BaseModel):
    """Full parsed screenplay representation."""
    metadata: ScreenplayMetadata
    scenes: List[SceneData]
    characters: List[CharacterRecord]
    raw_preview_text: str  # First ~5 pages of raw Courier text for preview
