# SymptomAssist AI: Neuro-symbolic Healthcare Diagnosis System

SymptomAssist is a hybrid AI medical advisor that combines **Symbolic Logic** (via Knowledge Graphs) and **Neural Networks** (via LLMs) to provide accurate, grounded, and empathetic health assessments.

---

##  Project Structure
The project the following architecture:

```text
cl_symptom/
├── app/                  # Main application package
│   ├── core/             # AI & Diagnostic logic
│   │   ├── __init__.py
│   │   ├── knowledge_graph.py   # Symbolic Inference (NetworkX)
│   │   ├── nlp_extractor.py     # Symptom extraction (Lexicon-based)
│   │   └── rag_pipeline.py      # Medical RAG (Semantic Retrieval)
│   ├── __init__.py
│   └── main.py           # FastAPI Web Server (Orchestration)
├── data/                 # Knowledge Datasets
│   ├── symptom_disease.csv
│   └── medical_docs.csv
├── static/               # Frontend Assets
│   └── index.html        # Premium Glassmorphism UI
├── .env                  # Environment Variables (API Keys)
├── requirements.txt      # Project Dependencies
├── CONTRIBUTING.md       # OSS Contributor Guidelines
└── README.md             # This file
```

---

##  Setup Instructions

### 1. Prerequisites
- **Python 3.10+** installed.
- **Groq API Key**: Obtain one from [Groq Cloud Console](https://console.groq.com/).

### 2. Install Dependencies
Run the following command in your terminal:
```bash
pip install -r requirements.txt
```

### 3. Configure Environment
Create/Edit the [`.env`](file:///.env) file at the project root:
```text
GEMINI_API_KEY=your_gemini_key_optional
GROQ_API_KEY=your_groq_api_key_here
```

---

##  Running the System

To start the FastAPI server with live-reloading:
```bash
python -m app.main
```
Once started, open your browser at [**http://127.0.0.1:8000**](http://127.0.0.1:8000).

---

##  Key Technical Features
1.  **Neuro-symbolic Reasoning**: Decouples medical facts from conversational reasoning (LLM).
2.  **Logic-Grounded RAG**: Injects curated medical documentation into the LLM prompt to eliminate hallucinations.
3.  **Real-time Diagnostics Dashboard**: Live visualization of extracted symptoms, KB matches, and RAG sources.
4.  **Priority Red Flag Detection**: Automatic highlighting of critical symptoms requiring emergency care.

---


