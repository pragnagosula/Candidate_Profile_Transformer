# Import all parsers so their @register decorators run at package import time.
# The registry is populated as a side-effect — no explicit registration call needed.
from app.parsers import csv_parser, json_parser, pdf_parser, txt_parser  # noqa: F401
from app.parsers.registry import parser_registry  # noqa: F401

__all__ = ["parser_registry"]
