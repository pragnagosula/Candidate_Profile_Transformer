#!/usr/bin/env python3
"""Entry point for the Eightfold candidate data pipeline.

Run with --help to see all options::

    python main.py --help
    python main.py --csv data/candidates.csv --output output/profiles.json
"""

from app.cli import main_cli

if __name__ == "__main__":
    main_cli()
