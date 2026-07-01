"""FastAPI application entrypoint for the Candidate Data Transformer."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.routes import router

app = FastAPI(
    title="Eightfold Candidate Data Transformer",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url=None,
)

app.mount("/static", StaticFiles(directory="static"), name="static")
app.include_router(router)
