"""Command-line interface for the Eightfold candidate data pipeline.

Usage examples::

    python main.py --csv data/candidates.csv --output output/profiles.json
    python main.py --csv a.csv --json b.json --pdf resume.pdf -o out.json
    python main.py --csv data/candidates.csv --verbose
    python main.py --csv data/candidates.csv --config custom/pipeline.yaml
"""

from __future__ import annotations

import sys

import click

from app.config.loader import get_config, reset_config_cache
from app.models.candidate import DataSource
from app.pipeline import Pipeline, PipelineResult
from app.utils.logging_config import get_logger

logger = get_logger(__name__)


@click.command("eightfold-pipeline")
@click.option(
    "--csv",
    "csv_files",
    multiple=True,
    metavar="FILE",
    help="CSV file containing candidate rows (repeatable).",
)
@click.option(
    "--json",
    "json_files",
    multiple=True,
    metavar="FILE",
    help="JSON file with one or more candidate records (repeatable).",
)
@click.option(
    "--pdf",
    "pdf_files",
    multiple=True,
    metavar="FILE",
    help="PDF resume file (repeatable).",
)
@click.option(
    "--txt",
    "txt_files",
    multiple=True,
    metavar="FILE",
    help="Plain-text resume file (repeatable).",
)
@click.option(
    "--output",
    "-o",
    default=None,
    metavar="FILE",
    help="Output JSON path (default: output/candidate_profile.json).",
)
@click.option(
    "--config",
    "-c",
    "config_path",
    default=None,
    metavar="FILE",
    help="Custom pipeline.yaml configuration file.",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    help="Enable DEBUG-level logging.",
)
@click.version_option(version="1.0.0", prog_name="eightfold-pipeline")
def main_cli(
    csv_files: tuple[str, ...],
    json_files: tuple[str, ...],
    pdf_files: tuple[str, ...],
    txt_files: tuple[str, ...],
    output: str | None,
    config_path: str | None,
    verbose: bool,
) -> None:
    """Eightfold Candidate Data Transformation Pipeline.

    Parses candidate data from CSV, JSON, and/or PDF sources,
    deduplicates and merges records for the same person, scores
    confidence, tracks provenance, and produces a validated JSON
    profile.

    \b
    Examples:
        python main.py --csv data/candidates.csv -o output/profiles.json
        python main.py --csv a.csv --json b.json --pdf resume.pdf
        python main.py --csv data/c.csv --txt resume.txt
        python main.py --csv data/c.csv --config custom/pipeline.yaml -v
    """
    if verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)

    inputs: list[tuple[DataSource, str]] = []
    for f in csv_files:
        inputs.append((DataSource.CSV, f))
    for f in json_files:
        inputs.append((DataSource.JSON, f))
    for f in pdf_files:
        inputs.append((DataSource.RESUME_PDF, f))
    for f in txt_files:
        inputs.append((DataSource.RESUME_TXT, f))

    if not inputs:
        click.echo(
            "Error: at least one input file is required.\n"
            "Use --csv, --json, or --pdf to specify input files.\n"
            "Run with --help for usage.",
            err=True,
        )
        sys.exit(1)

    if config_path:
        reset_config_cache()
        config = get_config(config_path)
        pipeline = Pipeline(config=config)
    else:
        pipeline = Pipeline()

    click.echo(f"Processing {len(inputs)} input file(s)...")

    result: PipelineResult = pipeline.run(inputs=inputs, output_path=output)

    click.echo(f"  Inputs parsed:     {result.total_inputs}")
    click.echo(f"  Candidate groups:  {result.total_groups}")
    click.echo(f"  Profiles produced: {len(result.profiles)}")

    if result.errors:
        click.echo(f"\nPipeline errors ({len(result.errors)}):", err=True)
        for err in result.errors:
            click.echo(f"  - {err}", err=True)
        sys.exit(1)

    if output:
        click.echo(f"Output written to: {output}")

    click.echo("Done.")
