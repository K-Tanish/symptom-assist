# SymptomAssist AI

SymptomAssist AI is a neuro-symbolic medical assistant that combines:
- Structured reasoning over a symptom-condition knowledge graph
- Retrieval-augmented context from curated medical documents
- Conversational response generation through an LLM

It is designed for educational triage support, not clinical diagnosis.

## Latest Updates

- Added accessible modal workflows for New Chat and Visit Summary dialogs
- Improved summary tooling with copy, print, and PDF export support
- Improved diagnostics UX with richer condition confidence and checklist views
- Strengthened graph visualization state handling and responsive rerender behavior
- Added better frontend structure via shared header/control styles

## Core Features

- Knowledge graph traversal using symptom-to-condition edges
- Red-flag symptom detection for urgent guidance
- RAG grounding from domain-specific medical documents
- Conversation memory via session timeline support
- Explainable diagnostics panel with confidence and contribution data
- Visit-ready summary generation for clinician handoff

## Architecture

The app uses a layered pipeline:
1. NLP extractor parses user symptom text.
2. Knowledge graph scores candidate conditions.
3. RAG pipeline retrieves matching medical context.
4. LLM composes grounded, user-friendly guidance.
5. Frontend renders diagnostics and graph state updates.

For a full architecture walkthrough, see [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md).

## Repository Structure

- `app/`: FastAPI backend and reasoning pipeline
- `data/`: CSV datasets and local vector store artifacts
- `static/`: Frontend HTML, CSS, and JavaScript
- `tests/`: Automated tests
- `scratch/`: Local experiments and debug scripts
- `.github/workflows/`: CI and automation

## Quick Start

### 1. Prerequisites

- Python 3.10+
- pip
- Groq API key

### 2. Clone and Install

```bash
git clone https://github.com/KGFCH2/symptom-assist.git
cd symptom-assist
pip install -r requirements.txt
```

### 3. Configure Environment

Create `.env` in project root:

```env
GROQ_API_KEY=your_groq_api_key
```

### 4. Run the App

```bash
uvicorn app.main:app --reload --port 8000
```

Open `http://localhost:8000`.

## API Endpoints

- `POST /chat`: Main chat and reasoning orchestration endpoint
- `GET /graph-data`: Graph data for D3 visualization
- `POST /debug/analyse`: Internal diagnostics payload for debugging
- `GET /summary/{session_id}`: Visit-ready summary payload
- `POST /session/clear`: Reset session timeline state

## Development Notes

- Keep `.env` and secrets out of version control
- Follow contribution conventions in [CONTRIBUTING.md](CONTRIBUTING.md)
- Respect security guidance in [Security.md](Security.md)
- Use focused, single-purpose pull requests

## Testing

Run test suite:

```bash
pytest -q
```

Optional static checks in repository artifacts:
- `pylint.txt`
- `bandit.json`

## Disclaimer

This project is for educational and research assistance only.
It does not provide medical diagnosis or treatment.
In emergencies, contact local emergency services immediately.
