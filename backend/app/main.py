"""
Story Analyzer AI - FastAPI Application Factory.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.api.v1.router import api_v1_router
from app.api.v1.endpoints.theory import get_retriever


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize the knowledge base retriever on startup
    get_retriever()
    yield


app = FastAPI(
    title="Story Analyzer AI API",
    description="Production-grade AI Screenplay Analysis Platform with Strict Evidence Grounding.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for local Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API routers
app.include_router(api_v1_router)

# Mount frontend static web application
from fastapi.staticfiles import StaticFiles
from pathlib import Path
frontend_dir = Path(__file__).resolve().parent.parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/app", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
    @app.get("/")
    def index():
        from fastapi.responses import FileResponse
        return FileResponse(str(frontend_dir / "index.html"))


@app.get("/health", tags=["System"])
def health_check():
    return {
        "status": "HEALTHY",
        "service": "Story Analyzer AI Backend",
        "version": "1.0.0"
    }

