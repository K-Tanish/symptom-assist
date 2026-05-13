# INSTRUCTIONS 📘

This document lists tracked project files with very brief working principles.

## Root Files 🧾

- `.env.example`: Template environment variables for local setup.
- `.gitignore`: Ignore rules for generated, secret, and local-only files.
- `bandit.json`: Security scan output/config artifact.
- `CODEOFCONDUCT.md`: Community behavior and participation policy.
- `CONTRIBUTING.md`: Contribution workflow, standards, and PR expectations.
- `License.md`: Repository licensing terms.
- `README.md`: Main project overview, setup, and usage guide.
- `requirements.txt`: Python dependency list for backend/runtime.
- `Security.md`: Security and responsible usage guidance.
- `SYSTEM_ARCHITECTURE.md`: Technical architecture and data-flow documentation.
- `error_log.txt`: Captured error output reference log.
- `pylint.txt`: Lint report snapshot.
- `pylint_utf8.txt`: UTF-8 lint report variant.
- `test_inputs.txt`: Sample input payloads for testing.
- `test_output.txt`: Sample expected/observed output.
- `INSTRUCTIONS.md`: File-by-file brief working principles (this file).

## GitHub Workflows ⚙️

- `.github/workflows/NsocLabeller.yml`: NSOC automation labeling workflow.
- `.github/workflows/tests.yml`: CI workflow for repository tests/checks.

## Backend Application 🖥️

- `app/__init__.py`: Package marker for backend module.
- `app/logging_config.py`: Logging initialization and handler configuration.
- `app/main.py`: FastAPI entrypoint and orchestration layer for chat/summary endpoints.

### Core Reasoning Modules 🧠

- `app/core/__init__.py`: Package marker for core logic.
- `app/core/error_handler.py`: Centralized API error classification and retry handling.
- `app/core/knowledge_graph.py`: Graph construction, traversal, condition scoring, red-flag logic.
- `app/core/nlp_extractor.py`: Symptom extraction from natural language text.
- `app/core/rag_pipeline.py`: Retrieval pipeline for medical context grounding.

## Frontend 🎨

- `static/index.html`: Main UI layout and modal/report template structure.
- `static/style.css`: Visual theme, layout, diagnostics panel, graph and modal styles.
- `static/app.js`: Frontend behavior, API calls, graph rendering, diagnostics updates, summary actions.

## Data 🗂️

- `data/medical_docs.csv`: Medical reference text corpus used by RAG.
- `data/symptom_disease.csv`: Symptom-to-condition mapping dataset.
- `data/chroma_db/chroma.sqlite3`: Local vector/index store artifact used by retrieval.

## Assets and Screenshots 🖼️

- `assets/Chat_and_Knowledge_graph.png`: README visual for chat and graph panel.
- `assets/diagnostics-panel.png`: README visual for diagnostics panel.
- `screenshots/before-markdown.png`: UI screenshot before markdown improvements.
- `screenshots/after-markdown.png`: UI screenshot after markdown improvements.

## Logs 📜

- `logs/error.log`: Runtime application error log output.

## Tests ✅

- `tests/__init__.py`: Tests package marker.
- `tests/test_graph_matching.py`: Validates graph matching/scoring behavior.
- `tests/test_ui_markdown.py`: Validates markdown/UI rendering-related behavior.

## Scratch / Local Experiments 🧪

- `scratch/debug_denies.py`: Debug helper for edge-case denial handling.
- `scratch/test_chunking.py`: Experimental script for chunking behavior checks.
- `scratch/test_temporal.py`: Experimental script for temporal symptom logic checks.
