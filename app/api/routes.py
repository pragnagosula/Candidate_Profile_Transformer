"""FastAPI routes for the Candidate Data Transformer web UI.

GET  /           → renders index.html (upload form)
POST /transform  → receives uploaded files, runs the existing Pipeline,
                   returns JSON for client-side rendering
"""

from __future__ import annotations

import asyncio
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.config.loader import get_config, reset_config_cache
from app.models.candidate import DataSource
from app.pipeline import Pipeline, PipelineResult

router = APIRouter()
templates = Jinja2Templates(directory="templates")

# File extension → DataSource routing.
_EXT_TO_SOURCE: dict[str, DataSource] = {
    ".csv":  DataSource.CSV,
    ".json": DataSource.JSON,
    ".pdf":  DataSource.RESUME_PDF,
    ".docx": DataSource.RESUME_PDF,
    ".txt":  DataSource.RESUME_TXT,
}


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


@router.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    # Starlette 1.x: TemplateResponse(request, name, context)
    return templates.TemplateResponse(request, "index.html")


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


@router.post("/transform")
async def transform(
    candidate_files: list[UploadFile] = File(...),
    config_file: Optional[UploadFile] = File(default=None),
) -> dict:
    """Accept uploaded candidate files, run the pipeline, return results.

    The Pipeline is called unchanged — no business logic lives here.
    """
    # Validate at least one real file was sent
    valid = [f for f in candidate_files if f.filename]
    if not valid:
        raise HTTPException(status_code=400, detail="At least one candidate file is required.")

    tmpdir = Path(tempfile.mkdtemp())
    try:
        inputs: list[tuple[DataSource, Path]] = []
        unsupported: list[str] = []

        for upload in valid:
            suffix = Path(upload.filename).suffix.lower()
            source = _EXT_TO_SOURCE.get(suffix)
            if source is None:
                unsupported.append(upload.filename)
                continue
            dest = tmpdir / upload.filename
            dest.write_bytes(await upload.read())
            inputs.append((source, dest))

        if not inputs:
            raise HTTPException(
                status_code=400,
                detail=f"No supported files provided. Unsupported: {unsupported}",
            )

        # Optional custom config
        config = None
        if config_file and config_file.filename:
            cfg_dest = tmpdir / config_file.filename
            cfg_dest.write_bytes(await config_file.read())
            reset_config_cache()
            config = get_config(str(cfg_dest))

        output_path = tmpdir / "output.json"

        # Run synchronous pipeline in a thread to keep the event loop free
        result: PipelineResult = await asyncio.to_thread(
            _run_pipeline, inputs, config, output_path
        )

        # Resolve the active projection config for the field-name map
        active_config = config or get_config()
        # output_field_map() returns {source_field: output_name}; invert it
        field_map = {v: k for k, v in active_config.projection.output_field_map().items()}

        return {
            "profiles": [p.model_dump(mode="json") for p in result.profiles],
            "field_map": field_map,
            "total_inputs": result.total_inputs,
            "total_groups": result.total_groups,
            "errors": result.errors,
            "success": result.success,
            "unsupported_files": unsupported,
        }

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_pipeline(
    inputs: list[tuple[DataSource, Path]],
    config,
    output_path: Path,
) -> PipelineResult:
    """Thin wrapper so Pipeline.run() can be called via asyncio.to_thread."""
    pipeline = Pipeline(config=config)
    return pipeline.run(inputs=inputs, output_path=output_path)
