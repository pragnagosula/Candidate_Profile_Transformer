"""FastAPI web server for the Eightfold Candidate Data Transformer.

Start with:
    uvicorn server:app --reload

Then open http://localhost:8000
"""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

app = FastAPI(
    title="Eightfold Candidate Data Transformer",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url=None,
)

app.mount("/static", StaticFiles(directory="static"), name="static")

from app.api.routes import router  # noqa: E402
app.include_router(router)
