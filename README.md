<h1 align="center">Candidate Profile Transformer</h1>

<p align="center">
  <em>Transform raw candidate data from any source into a single, structured, confidence-scored profile.</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-0.109+-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/Pydantic-v2-E92063?style=for-the-badge&logo=pydantic&logoColor=white" />
  <img src="https://img.shields.io/badge/Tests-951%20passing-4CAF50?style=for-the-badge&logo=pytest&logoColor=white" />
  <img src="https://img.shields.io/badge/License-Academic-blue?style=for-the-badge" />
</p>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Folder Structure](#-folder-structure)
- [Tech Stack](#-tech-stack)
- [Installation](#-installation)
- [Running the Project](#-running-the-project)
- [Supported Input Formats](#-supported-input-formats)
- [Expected Output](#-expected-output)
- [Confidence Engine](#-confidence-engine)
- [Provenance Engine](#-provenance-engine)
- [Validation Engine](#-validation-engine)
- [Screenshots](#-screenshots)
- [Future Improvements](#-future-improvements)
- [Author](#-author)
- [License](#-license)

---

## 🌟 Overview

**Candidate Profile Transformer** is a production-grade data pipeline that ingests candidate information from heterogeneous sources — PDF resumes, CSV ATS exports, JSON API payloads, and plain-text resumes — and fuses them into a single, validated, confidence-scored **Unified Candidate Profile**.

### The Problem

Recruitment data is messy and fragmented. A single candidate may appear as:

- A PDF resume uploaded directly
- A row in a bulk CSV export from an ATS (Applicant Tracking System)
- A JSON payload from a LinkedIn scraper or HR API
- A plain-text resume emailed in

Each source spells field names differently, formats dates inconsistently, lists skills in different orders, and may have errors or missing fields. Manually reconciling this data is error-prone and slow.

### The Solution

This pipeline automates that reconciliation end-to-end:

1. **Parses** every source format into a common raw structure
2. **Extracts** typed fields (name, email, skills, experience, education, links) using format-specific rules and alias resolution
3. **Normalizes** values to canonical forms (E.164 phone numbers, ISO dates, skill name aliases)
4. **Validates** each record against configurable field-level rules
5. **Resolves** which records across sources belong to the same real person (using email exact-match + fuzzy name matching via a Union-Find algorithm)
6. **Merges** records from multiple sources using strategy-driven rules (priority, most-complete, union)
7. **Scores** the merged result with a weighted confidence report
8. **Tracks provenance** — every field value knows which source it came from, what it was before normalization, and how confident we are in it
9. **Projects** the final profile into a clean JSON schema and writes the output

---

## ✨ Key Features

| Category | Feature |
|---|---|
| **Ingestion** | PDF Resume Parsing, CSV ATS Parsing, JSON API Parsing, TXT Resume Parsing |
| **Extraction** | Name, Email, Phone, Location, Summary, Skills, Experience, Education, Projects, Certifications, Achievements, Languages, Links |
| **Robustness** | 100+ field alias resolution, nested JSON unwrapping, skill name normalization, multi-separator parsing |
| **Intelligence** | Entity Resolution (Union-Find + RapidFuzz), multi-source merging, fuzzy deduplication of experience and education lists |
| **Quality** | 10-factor confidence scoring, full provenance tracking, 7-validator validation engine, schema validation |
| **Interface** | FastAPI web UI, REST API endpoint, Click-based CLI |
| **Testing** | 951 unit tests, 0 failures |

---

## 🏗 System Architecture

The pipeline is fully sequential and stateless between runs. Every layer communicates through typed Pydantic models — no raw dicts cross layer boundaries.

```
┌─────────────────────────────────────────────────┐
│                  Input Files                    │
│       .pdf   .csv   .json   .txt                │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│                  Parser Layer                   │
│  PDFParser │ CSVParser │ JSONParser │ TXTParser  │
│  → RawCandidateData (one per candidate record)  │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│                 Extractor Layer                 │
│  PDFExtractor │ CSVExtractor │ JSONExtractor    │
│  TXTExtractor                                   │
│  → ExtractedCandidate (typed, structured)       │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│              Normalization Engine               │
│  Name │ Email │ Phone │ Date │ Skill │ URL      │
│  → NormalizedCandidate + NormalizationDiff      │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│               Validation Engine                 │
│  Email │ Phone │ Date │ URL │ Skills            │
│  Required Fields │ Completeness                 │
│  → ValidationResult (errors + warnings)         │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│              Entity Resolver                    │
│  Email exact-match + Name fuzzy-match           │
│  Union-Find algorithm (O(n²), path-halving)     │
│  → CandidateGroup[] (same person across sources)│
└────────────────────┬────────────────────────────┘
                     │  (per group)
                     ▼
┌─────────────────────────────────────────────────┐
│               Merge Engine                      │
│  Strategies: priority │ most_complete │ union   │
│  Fuzzy dedup for experience + education lists   │
│  → MergedCandidate                              │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│             Confidence Engine                   │
│  10-factor weighted scoring model               │
│  Source reliability × recency × extraction     │
│  confidence × cross-field validation            │
│  → ConfidenceReport (overall + per-field)       │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│              Provenance Engine                  │
│  Per-field lineage: source, original value,     │
│  normalized value, method, confidence, notes    │
│  → list[ProvenanceEntry]                        │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│          Projection + Schema Validation         │
│  Config-driven field selection + renaming       │
│  Final schema check                             │
│  → CandidateProfile (JSON-ready)                │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│               JSON Output File                  │
│         output/candidate_profiles.json          │
└─────────────────────────────────────────────────┘
```

---

## 📁 Folder Structure

```
eightfold/
│
├── app/
│   ├── api/                       # FastAPI routes + web UI handlers
│   │   └── routes.py
│   │
│   ├── parsers/                   # Format-specific file parsers
│   │   ├── pdf_parser.py          # pdfplumber-based PDF parser
│   │   ├── csv_parser.py          # CSV row → RawCandidateData
│   │   ├── json_parser.py         # JSON object/array → RawCandidateData
│   │   ├── txt_parser.py          # Plain-text parser
│   │   ├── base.py                # BaseParser ABC
│   │   └── registry.py           # Auto-dispatch by DataSource
│   │
│   ├── extractors/                # Typed field extraction per format
│   │   ├── pdf_extractor.py       # Section detection, regex, URL extraction
│   │   ├── csv_extractor.py       # Column alias resolution, flat sub-fields
│   │   ├── json_extractor.py      # Nested unwrapping, recursive field search
│   │   ├── txt_extractor.py       # Same pipeline as PDF over raw text
│   │   ├── text_resume_parser.py  # Core NLP: section parsing, experience/education
│   │   ├── field_map.py           # 100+ field alias table (canonical → aliases)
│   │   ├── base.py                # BaseExtractor ABC
│   │   └── registry.py           # Auto-dispatch by DataSource
│   │
│   ├── normalizers/               # Canonical-form normalizers
│   │   ├── engine.py              # Orchestrates all normalizers, tracks diffs
│   │   ├── name_normalizer.py
│   │   ├── email_normalizer.py
│   │   ├── phone_normalizer.py    # E.164 via phonenumbers library
│   │   ├── date_normalizer.py     # ISO 8601 via python-dateutil
│   │   ├── skill_normalizer.py    # Alias map + casing rules
│   │   └── url_normalizer.py      # Scheme normalization
│   │
│   ├── validators/                # Field-level validation rules
│   │   ├── engine.py              # Aggregator: runs all validators
│   │   ├── email_validator.py
│   │   ├── phone_validator.py
│   │   ├── date_validator.py
│   │   ├── url_validator.py
│   │   ├── skills_validator.py
│   │   ├── required_fields.py
│   │   └── completeness_validator.py
│   │
│   ├── mergers/                   # Entity resolution + merging
│   │   ├── entity_resolver.py     # Union-Find + email/name matching
│   │   ├── merge_engine.py        # Strategy-driven field + list merging
│   │   └── candidate_group.py     # CandidateGroup container
│   │
│   ├── confidence/
│   │   └── engine.py              # 10-factor confidence scoring
│   │
│   ├── provenance/
│   │   └── engine.py              # Per-field lineage tracking
│   │
│   ├── projection/
│   │   └── engine.py              # Config-driven field selection + renaming
│   │
│   ├── schema/
│   │   └── validator.py           # Final JSON schema validation
│   │
│   ├── pipeline/
│   │   └── orchestrator.py        # Pipeline.run() — wires all layers
│   │
│   ├── models/
│   │   └── candidate.py           # All Pydantic domain models
│   │
│   ├── config/                    # Config loading + Pydantic YAML models
│   └── utils/
│       └── logging_config.py
│
├── config/                        # YAML configuration files
│   ├── pipeline.yaml              # Output directory + filename
│   ├── merge_rules.yaml           # Per-field merge strategies
│   ├── entity_resolution.yaml     # Name fuzzy threshold, email match flag
│   ├── normalization.yaml         # Normalizer feature flags
│   ├── projection.yaml            # Output field selection + renaming
│   └── source_reliability.yaml   # Per-source trust weights (0.0–1.0)
│
├── tests/
│   └── unit/                      # 951 unit tests, 0 failures
│
├── templates/
│   └── index.html                 # Jinja2 web UI template
│
├── static/                        # CSS + JS for web UI
├── samples/                       # Sample input files (PDF, CSV, JSON)
├── output/                        # Generated profiles written here
├── main.py                        # FastAPI application entry point
└── requirements.txt
```

---

## 🛠 Tech Stack

### Backend

| Technology | Version | Purpose |
|---|---|---|
| **Python** | 3.11+ | Core language |
| **FastAPI** | ≥ 0.109 | REST API + web UI server |
| **Uvicorn** | ≥ 0.27 | ASGI server |
| **Pydantic** | v2 | Domain models, validation, serialization |
| **Click** | ≥ 8.1 | Command-line interface |
| **Rich** | ≥ 13.7 | Terminal formatting for CLI output |

### Parsing & Extraction

| Library | Version | Purpose |
|---|---|---|
| **pdfplumber** | ≥ 0.10 | PDF text + layout extraction |
| **pypdf** | ≥ 3.17 | PDF metadata + annotation link extraction |
| **python-dateutil** | ≥ 2.8 | Flexible date string parsing |
| **phonenumbers** | ≥ 8.13 | E.164 phone number normalization |

### Intelligence & Matching

| Library | Version | Purpose |
|---|---|---|
| **RapidFuzz** | ≥ 3.6 | Fuzzy string matching for entity resolution and list deduplication |

### Configuration & Quality

| Tool | Purpose |
|---|---|
| **PyYAML** | YAML-based pipeline configuration |
| **pytest + pytest-cov + pytest-mock** | Unit testing (951 tests) |
| **mypy** | Static type checking |
| **ruff** | Linting + code formatting |

### Frontend

| Technology | Purpose |
|---|---|
| **Jinja2** | HTML templating (bundled with FastAPI/Starlette) |
| **HTML / CSS / JavaScript** | Drag-and-drop upload UI + profile result display |

---

## ⚙️ Installation

### Prerequisites

- Python **3.11** or higher
- `pip`

### Steps

**1. Clone the repository**

```bash
https://github.com/pragnagosula/Candidate_Profile_Transformer.git
cd Candidate_Profile_Transformer
```

**2. Create a virtual environment**

```bash
python -m venv venv
```

**3. Activate the virtual environment**

On **Windows**:
```bash
venv\Scripts\activate
```

On **macOS / Linux**:
```bash
source venv/bin/activate
```

**4. Install dependencies**

```bash
pip install -r requirements.txt
```

**5. Verify the installation**

```bash
python -m pytest -q
```

You should see `951 passed` with no failures.

---

## 🚀 Running the Project

### Web Application (Recommended)

Start the FastAPI development server:

```bash
uvicorn app.main:app --reload
```

This command:
- Starts the ASGI server on **http://127.0.0.1:8000**
- Enables **hot-reload** — the server restarts automatically when you save a file
- Serves the drag-and-drop web UI at the root URL
- Exposes the `/transform` REST endpoint for programmatic access

Once running, open your browser and go to:

```
http://127.0.0.1:8000
```

Upload one or more candidate files (`.pdf`, `.csv`, `.json`, `.txt`) and click **Transform**. The unified candidate profile appears on the page and is simultaneously written to `output/candidate_profiles.json`.

### REST API

You can call the pipeline directly without the browser UI:

```bash
curl -X POST http://127.0.0.1:8000/transform \
  -F "candidate_files=@samples/GMSPRAGNA_CV.pdf" \
  -F "candidate_files=@samples/candidates.csv"
```

### Interactive API Docs

FastAPI generates interactive documentation automatically:

| URL | Interface |
|---|---|
| `http://127.0.0.1:8000/docs` | Swagger UI — try the API in the browser |
| `http://127.0.0.1:8000/redoc` | ReDoc — clean reference documentation |

---

## 📂 Supported Input Formats

| Format | Extension | Extractor | Typical Source |
|---|---|---|---|
| **PDF Resume** | `.pdf` | `PDFExtractor` | Candidate direct upload, emailed resume |
| **CSV Export** | `.csv` | `CSVExtractor` | ATS bulk export, HR spreadsheet |
| **JSON Payload** | `.json` | `JSONExtractor` | LinkedIn API, HR system REST API |
| **Plain Text** | `.txt` | `TXTExtractor` | Pasted resume text, legacy systems |

### Format Flexibility

#### JSON — Any Nesting Depth

The JSON extractor unwraps common wrapper structures automatically — flat, single-wrapper, and double-wrapper inputs all produce identical output:

```json
// Flat
{ "name": "Alice", "email": "alice@example.com" }

// Wrapped in "candidate"
{ "candidate": { "name": "Alice", "email": "alice@example.com" } }

// Nested data.candidate
{ "data": { "candidate": { "name": "Alice" } } }
```

Nested contact sections are also handled:

```json
{
  "name": "Alice",
  "contact": { "email": "alice@example.com", "phone": "+91 9876543210" }
}
```

#### CSV — 100+ Column Name Aliases

`Full Name`, `Applicant Name`, `Candidate Name`, `full_name` all resolve to the same `name` field. The skills cell accepts comma, semicolon, pipe, newline, and JSON array formats interchangeably.

---

## 📊 Confidence Engine

The confidence engine produces a **10-factor weighted score** (0.0 – 1.0) for every merged candidate.

### Score Formula

```
overall = w_field        × weighted_field_average
        + w_completeness × completeness
        + w_agreement    × source_agreement
        − validation_penalty
```

Where `(w_field, w_completeness, w_agreement)` are **adaptive** — they scale with the number of contributing sources.

### Scoring Factors

| # | Factor | Description |
|---|---|---|
| 1 | **Source reliability** | Per-source trust weight from `source_reliability.yaml` (e.g. PDF > JSON) |
| 2 | **Field importance** | Per-field weights — email/name count more than location |
| 3 | **Completeness** | Fraction of required + recommended fields present |
| 4 | **Source agreement** | Fuzzy similarity between sources on shared fields (RapidFuzz) |
| 5 | **Conflict penalty** | Deducted when two sources actively disagree on the same field |
| 6 | **Agreement bonus** | Proportional bonus when sources confirm each other |
| 7 | **Recency multiplier** | More recently updated sources score higher |
| 8 | **Extraction confidence** | OCR / parser quality multiplier |
| 9 | **Cross-field validation** | Email format check, LinkedIn/name mismatch detection |
| 10 | **Adaptive weights** | Score weights scale with number of contributing sources |

All thresholds and weights are fully configurable in `config/source_reliability.yaml`.

---

## 🔍 Provenance Engine

Every field value in the final profile carries a complete **lineage record** answering:

| Question | Provenance Field |
|---|---|
| Where did this come from? | `source: "resume_pdf"` |
| What was the raw value? | `original_value: "+91 98765 43210"` |
| What is the canonical form? | `normalized_value: "+919876543210"` |
| How was it extracted? | `extraction_method: "regex"` / `"direct"` / `"inferred"` |
| How confident are we? | `confidence: 0.9` |
| What changed and why? | `notes: "Normalized to E.164 format"` |

This makes every output field fully **auditable** — you can trace any value back to its source file, the raw string it came from, and every transformation applied to it.

---

## ✅ Validation Engine

Seven independent validators run in sequence on every normalized candidate. A buggy validator cannot crash the pipeline or suppress results from others.

| Validator | What It Checks |
|---|---|
| **EmailValidator** | RFC-5321 format, valid domain structure |
| **PhoneValidator** | Parseable by the `phonenumbers` library, valid country code |
| **DateValidator** | Chronological order (start ≤ end), no impossible future graduation dates |
| **URLValidator** | HTTP/HTTPS scheme, structurally valid URL |
| **SkillsValidator** | Minimum skill count, no obviously non-skill strings |
| **RequiredFieldsValidator** | Presence of name + email (configurable) |
| **CompletenessValidator** | Weighted scoring across required / recommended / optional fields |

Issues are classified as **error** (makes `is_valid = false`) or **warning** / **info** (informational only). Duplicate issues across multiple source records are deduplicated before the final result is assembled.

---

## 👩‍💻 Author

**Gosula Mohana Sree Pragna**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](https://linkedin.com/in/pragnagosula)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/pragnagosula)

---


