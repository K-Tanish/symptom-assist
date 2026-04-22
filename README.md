# SymptomAssist

Most symptom checkers give you a list of diseases and leave you to figure out the rest. SymptomAssist is different — it reasons through your symptoms the way a doctor would, combining structured medical knowledge with the conversational ability of modern LLMs.

The result is an AI health advisor that can explain *why* it suspects something, not just *what* it suspects.

---

## How it works

SymptomAssist uses a neuro-symbolic architecture — a design that keeps medical facts separate from language generation:

- **Knowledge Graph** (NetworkX): encodes relationships between symptoms, conditions, and red flags as structured logic. This is the "memory" of the system.
- **NLP Extractor**: reads what the user types and pulls out the symptoms being described, even when phrased casually.
- **RAG Pipeline**: retrieves relevant excerpts from curated medical documents and injects them into the LLM prompt — so responses are grounded in real sources, not hallucinated.
- **LLM (Groq)**: handles the conversational layer, generating empathetic and readable responses based on what the knowledge graph and RAG pipeline surface.

This separation matters. The LLM never "makes up" medical facts — it only interprets and communicates what the symbolic layer has already verified.

---

## What you get

- **Grounded assessments** — every diagnosis suggestion traces back to a knowledge base entry or retrieved document
- **Red flag detection** — critical symptoms (chest pain, difficulty breathing, etc.) are automatically flagged and surfaced before anything else
- **Live diagnostics panel** — see in real time which symptoms were extracted, which KB entries matched, and which documents were retrieved
- **Clean UI** — a glassmorphism interface that doesn't feel clinical

---

## Getting started

### Prerequisites

- Python 3.10+
- A [Groq API key](https://console.groq.com)

### Install

```bash
git clone https://github.com/K-Tanish/symptom-assist.git
cd symptom-assist
pip install -r requirements.txt
```

### Configure

Create a `.env` file at the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
GEMINI_API_KEY=your_gemini_key_optional
```

Or copy the example file:

```bash
cp .env.example .env
```

### Run

```bash
python -m app.main
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

---

## Project structure

```
cl_symptom/
├── app/
│   ├── core/
│   │   ├── knowledge_graph.py   # Symbolic inference (NetworkX)
│   │   ├── nlp_extractor.py     # Symptom extraction
│   │   └── rag_pipeline.py      # Medical RAG pipeline
│   └── main.py                  # FastAPI server
├── data/
│   ├── symptom_disease.csv
│   └── medical_docs.csv
├── static/
│   └── index.html               # Frontend
├── .env
├── requirements.txt
└── CONTRIBUTING.md
```

---

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines on how to get started — whether that's improving the knowledge base, extending the NLP extractor, or refining the UI.

---

## Disclaimer

SymptomAssist is not a substitute for professional medical advice. It is an educational and research tool. Always consult a qualified healthcare provider for medical decisions.