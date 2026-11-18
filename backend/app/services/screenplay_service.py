"""
Screenplay Service & Secure Storage Manager.
Handles file validation, cryptographic hashing, immutable raw file storage,
multi-format parsing, scene extraction, character registry generation, and persistence.
"""

import os
import shutil
import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional

from app.schemas.screenplay import (
    ScreenplayMetadata,
    SceneData,
    CharacterRecord,
    ScreenplayParsedResponse
)
from app.parsers.screenplay_parser import ScreenplayParser
from app.parsers.fountain_parser import FountainParser
from app.rag.extractors import DocumentExtractor
from app.engines.scene_engine import SceneEngine
from app.engines.character_engine import CharacterEngine


class ScreenplayService:
    """Core domain service for screenplay ingestion, parsing, and storage."""

    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB
    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".fountain"}

    def __init__(self, base_storage_dir: Optional[str] = None):
        if base_storage_dir is None:
            backend_root = Path(__file__).resolve().parent.parent.parent
            self.storage_dir = backend_root / "storage"
        else:
            self.storage_dir = Path(base_storage_dir)

        self.uploads_dir = self.storage_dir / "uploads"
        self.parsed_dir = self.storage_dir / "screenplays"
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.parsed_dir.mkdir(parents=True, exist_ok=True)

        # In-memory index of ingested screenplays
        self._screenplay_cache: Dict[str, ScreenplayParsedResponse] = {}
        self._load_persisted_index()

    def _load_persisted_index(self):
        """Loads cached screenplay metadata from disk on startup."""
        for json_file in self.parsed_dir.glob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                parsed = ScreenplayParsedResponse(**data)
                self._screenplay_cache[parsed.metadata.screenplay_id] = parsed
            except Exception:
                pass

    def validate_file(self, filename: str, file_size: int, content_bytes: bytes):
        """Validates file extension, size, and header signatures."""
        ext = Path(filename).suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported format '{ext}'. Supported formats: {', '.join(sorted(self.SUPPORTED_EXTENSIONS))}"
            )

        if file_size <= 0:
            raise ValueError("Uploaded file is empty (0 bytes).")

        if file_size > self.MAX_FILE_SIZE:
            raise ValueError(f"File size exceeds maximum allowed limit of {self.MAX_FILE_SIZE // (1024 * 1024)} MB.")

        # Magic byte validation
        if ext == ".pdf" and not content_bytes.startswith(b"%PDF"):
            raise ValueError("File has .pdf extension but invalid PDF header signature.")
        if ext == ".docx" and not content_bytes.startswith(b"PK\x03\x04"):
            raise ValueError("File has .docx extension but invalid Office Open XML ZIP header.")

    def ingest_screenplay(
        self,
        filename: str,
        content_bytes: bytes,
        title: Optional[str] = None,
        author: Optional[str] = None
    ) -> ScreenplayParsedResponse:
        """
        Securely stores original raw document and executes full parsing pipeline.
        Preserves original document completely without rewriting or modifying.
        """
        # 1. Validate
        file_size = len(content_bytes)
        self.validate_file(filename, file_size, content_bytes)

        # 2. Compute SHA-256 hash for deduplication and provenance
        file_hash = hashlib.sha256(content_bytes).hexdigest()
        screenplay_id = str(uuid.uuid4())
        ext = Path(filename).suffix.lower()

        # 3. Securely store original file (immutable preservation)
        stored_filename = f"{screenplay_id}{ext}"
        stored_path = self.uploads_dir / stored_filename
        with open(stored_path, "wb") as f:
            f.write(content_bytes)

        # 4. Extract raw text
        script_title = title or Path(filename).stem.replace("_", " ").title()
        script_author = author or "Unknown"

        if ext == ".fountain":
            raw_text = content_bytes.decode("utf-8", errors="replace")
            page_count = max(1, (len(raw_text.splitlines()) + 54) // 55)
            elements = FountainParser.parse_text(raw_text)
        elif ext in [".pdf", ".docx", ".txt"]:
            extracted = DocumentExtractor.extract_file(stored_path)
            raw_text = extracted.raw_text
            page_count = extracted.total_pages
            elements = ScreenplayParser.parse_document_text(raw_text)
        else:
            raise ValueError(f"Unsupported format: {ext}")

        # 5. Extract structured scenes
        scenes = SceneEngine.extract_scenes(elements)

        # 6. Extract character registry
        characters = CharacterEngine.build_registry(scenes)

        # 7. Generate Courier preview text (first ~5000 characters)
        preview_text = "\n".join(raw_text.splitlines()[:150])

        metadata = ScreenplayMetadata(
            screenplay_id=screenplay_id,
            title=script_title,
            author=script_author,
            format=ext.upper().replace(".", ""),
            file_size_bytes=file_size,
            file_hash=file_hash,
            page_count=page_count,
            scene_count=len(scenes),
            character_count=len(characters),
            status="PARSED",
            created_at=datetime.now(timezone.utc).isoformat()
        )

        response = ScreenplayParsedResponse(
            metadata=metadata,
            scenes=scenes,
            characters=characters,
            raw_preview_text=preview_text
        )

        # 8. Persist structured parsed JSON
        json_path = self.parsed_dir / f"{screenplay_id}.json"
        with open(json_path, "w", encoding="utf-8") as fp:
            json.dump(response.model_dump(), fp, indent=2)

        self._screenplay_cache[screenplay_id] = response
        return response

    def get_screenplay(self, screenplay_id: str) -> Optional[ScreenplayParsedResponse]:
        """Fetches full parsed screenplay by ID."""
        return self._screenplay_cache.get(screenplay_id)

    def list_screenplays(self) -> List[ScreenplayMetadata]:
        """Lists all uploaded screenplays."""
        return [item.metadata for item in self._screenplay_cache.values()]

    def get_scenes(self, screenplay_id: str) -> List[SceneData]:
        """Returns scenes for a screenplay."""
        script = self.get_screenplay(screenplay_id)
        return script.scenes if script else []

    def get_characters(self, screenplay_id: str) -> List[CharacterRecord]:
        """Returns characters for a screenplay."""
        script = self.get_screenplay(screenplay_id)
        return script.characters if script else []
