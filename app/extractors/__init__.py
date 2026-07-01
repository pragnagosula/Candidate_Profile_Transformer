# Trigger self-registration of all extractors at package import time.
from app.extractors import csv_extractor, json_extractor, pdf_extractor, txt_extractor  # noqa: F401
from app.extractors.registry import extractor_registry  # noqa: F401

__all__ = ["extractor_registry"]
