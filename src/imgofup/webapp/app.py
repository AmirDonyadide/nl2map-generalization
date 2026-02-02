# src/imgofup/webapp/app.py
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from imgofup.webapp.api import create_api_router

# -----------------------------------------------------------------------------
# Paths (repo-aware)
# -----------------------------------------------------------------------------
# This file is: <repo>/src/imgofup/webapp/app.py
REPO_ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = REPO_ROOT / "models"

# ✅ Frontend lives inside the package:
# <repo>/src/imgofup/webapp/frontend
FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"

# -----------------------------------------------------------------------------
# App
# -----------------------------------------------------------------------------
app = FastAPI(
    title="IMGOFUP Local Generalization Demo",
    version="0.1.0",
)

# CORS: allow local frontend (file://) + local dev servers.
# If you serve frontend via FastAPI, CORS becomes mostly irrelevant.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1",
        "http://127.0.0.1:8000",
        "null",  # some browsers use "null" origin for file://
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes (defined in api.py)
app.include_router(create_api_router(MODELS_DIR))

if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

# -----------------------------------------------------------------------------
# Local run
# -----------------------------------------------------------------------------
def _print_startup_info() -> None:
    print("IMGOFUP Local Web App")
    print(f"- Repo root:      {REPO_ROOT}")
    print(f"- Models dir:     {MODELS_DIR}")
    if FRONTEND_DIR.exists():
        print(f"- Frontend dir:   {FRONTEND_DIR} (served at /)")
    else:
        print(f"- Frontend dir:   {FRONTEND_DIR} (NOT found; API only)")
    print("- API endpoints:  /api/models, /api/predict, /api/generalize")


if __name__ == "__main__":
    _print_startup_info()
    import uvicorn

    uvicorn.run(
        "imgofup.webapp.app:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )
