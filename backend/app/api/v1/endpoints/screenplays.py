"""
Screenplay Upload, Parsing, and Inspection API Endpoints.
Supports single and multiple file uploads (PDF, DOCX, TXT, Fountain),
validation, scene breakdowns, character registries, and raw Courier previews.
"""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends, Query, status

from app.schemas.screenplay import (
    ScreenplayMetadata,
    SceneData,
    CharacterRecord,
    ScreenplayParsedResponse
)
from app.services.screenplay_service import ScreenplayService

router = APIRouter(prefix="/screenplays", tags=["Screenplays & Parsing"])

_screenplay_service_instance: Optional[ScreenplayService] = None


def get_screenplay_service() -> ScreenplayService:
    global _screenplay_service_instance
    if _screenplay_service_instance is None:
        _screenplay_service_instance = ScreenplayService()
    return _screenplay_service_instance


@router.post("/upload", response_model=List[ScreenplayParsedResponse], summary="Upload and parse screenplays (Single or Multiple)")
async def upload_screenplays(
    files: List[UploadFile] = File(..., description="Screenplay files (.pdf, .docx, .txt, .fountain)"),
    title: Optional[str] = Form(None, description="Optional title override (for single upload)"),
    author: Optional[str] = Form(None, description="Optional author override"),
    service: ScreenplayService = Depends(get_screenplay_service)
) -> List[ScreenplayParsedResponse]:
    """
    Uploads and parses one or more screenplays.
    Performs format validation, SHA-256 deduplication, scene extraction,
    and character registry generation. Original documents are immutably preserved.
    """
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files provided in upload request."
        )

    parsed_results = []
    errors = []

    for file in files:
        try:
            content_bytes = await file.read()
            # If multiple files, use individual file titles
            file_title = title if (len(files) == 1 and title) else None
            parsed = service.ingest_screenplay(
                filename=file.filename or "untitled.txt",
                content_bytes=content_bytes,
                title=file_title,
                author=author
            )
            parsed_results.append(parsed)
        except ValueError as ve:
            errors.append(f"File '{file.filename}': {str(ve)}")
        except Exception as e:
            errors.append(f"File '{file.filename}': Failed to process screenplay ({str(e)})")

    if not parsed_results and errors:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"message": "All uploaded files failed validation.", "errors": errors}
        )

    return parsed_results


@router.get("", response_model=List[ScreenplayMetadata], summary="List all ingested screenplays")
def list_screenplays(
    service: ScreenplayService = Depends(get_screenplay_service)
) -> List[ScreenplayMetadata]:
    """Lists summary metadata of all screenplays currently stored in the system."""
    return service.list_screenplays()


@router.get("/{screenplay_id}", response_model=ScreenplayParsedResponse, summary="Get full parsed screenplay")
def get_screenplay_detail(
    screenplay_id: str,
    service: ScreenplayService = Depends(get_screenplay_service)
) -> ScreenplayParsedResponse:
    """Retrieves full parsed screenplay with scenes, character registry, and preview."""
    script = service.get_screenplay(screenplay_id)
    if not script:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Screenplay with ID '{screenplay_id}' not found."
        )
    return script


@router.get("/{screenplay_id}/scenes", response_model=List[SceneData], summary="Get scene breakdown")
def get_screenplay_scenes(
    screenplay_id: str,
    character_filter: Optional[str] = Query(None, description="Filter scenes containing character"),
    service: ScreenplayService = Depends(get_screenplay_service)
) -> List[SceneData]:
    """Returns structured scene breakdown for the screenplay with optional character filtering."""
    scenes = service.get_scenes(screenplay_id)
    if not scenes and not service.get_screenplay(screenplay_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Screenplay with ID '{screenplay_id}' not found."
        )

    if character_filter:
        char_clean = character_filter.strip().upper()
        scenes = [s for s in scenes if char_clean in s.characters_present]

    return scenes


@router.get("/{screenplay_id}/characters", response_model=List[CharacterRecord], summary="Get character registry")
def get_screenplay_characters(
    screenplay_id: str,
    service: ScreenplayService = Depends(get_screenplay_service)
) -> List[CharacterRecord]:
    """Returns character registry with appearances, dialogue counts, and relationship matrix."""
    chars = service.get_characters(screenplay_id)
    if not chars and not service.get_screenplay(screenplay_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Screenplay with ID '{screenplay_id}' not found."
        )
    return chars


@router.get("/{screenplay_id}/preview", summary="Get raw Courier screenplay preview text")
def get_screenplay_preview(
    screenplay_id: str,
    service: ScreenplayService = Depends(get_screenplay_service)
):
    """Returns the raw Courier formatted script text for preview rendering."""
    script = service.get_screenplay(screenplay_id)
    if not script:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Screenplay with ID '{screenplay_id}' not found."
        )
    return {
        "screenplay_id": screenplay_id,
        "title": script.metadata.title,
        "format": script.metadata.format,
        "raw_preview_text": script.raw_preview_text
    }
