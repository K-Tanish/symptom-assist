from .core.knowledge_graph import (
    load_graph_from_csv, traverse_graph, find_candidate_conditions,
    get_followup_questions, get_treatment, check_red_flags, graph_summary,
    rank_conditions_with_temporal_context   # ✅ ADD THIS
)"""
main.py
-------
FastAPI backend for the AI-powered symptom chatbot.

Architecture:
  1. User message arrives at POST /chat
  2. NLP extractor pulls symptom keywords from text  (dynamic lexicon from CSV)
  3. Knowledge Graph built from CSV; BFS traversal finds candidate conditions
  4. RAG pipeline retrieves relevant medical documents  (loaded from CSV)
  5. All context is injected into the LLM prompt
  6. LLM responds grounded in the retrieved medical knowledge

Dataset-driven: conditions, symptoms, and documents all come from
  data/symptom_disease.csv  and  data/medical_docs.csv
"""

import os
import json
import pathlib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
from groq import Groq
from dotenv import load_dotenv

from .core.knowledge_graph import (
    
    load_graph_from_csv, traverse_graph, find_candidate_conditions,
    get_followup_questions, get_treatment, check_red_flags, graph_summary,
    rank_conditions_with_temporal_context   #  ADD THIS
)
)
from .core.rag_pipeline import RAGPipeline
from .core.nlp_extractor import SymptomExtractor

load_dotenv()
---------------------------------------------------------------------
# Resolve dataset paths (relative to the project root)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = pathlib.Path(__file__).parent.parent
_SYMPTOM_CSV  = str(_PROJECT_ROOT / "data" / "symptom_disease.csv")
_DOCS_CSV     = str(_PROJECT_ROOT / "data" / "medical_docs.csv")

# ---------------------------------------------------------------------------
# Initialise AI components at startup
# ---------------------------------------------------------------------------

print("[startup] Building knowledge graph from CSV...")
GRAPH = load_graph_from_csv(_SYMPTOM_CSV)
print("[startup] Initialising RAG pipeline from CSV...")
RAG = RAGPipeline(csv_path=_DOCS_CSV)
print("[startup] Loading NLP extractor (dynamic lexicon from CSV)...")
NLP = SymptomExtractor(csv_path=_SYMPTOM_CSV)
print("[startup] Groq client ready.")
GROQ = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="SymptomAssist AI", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class Message(BaseModel):
    role: str       # "user" or "model" (Gemini uses "model", but frontend might still send it)
    content: str
    # New fields for temporal context
    duration: Optional[str] = None       # e.g., "3 days", "2 weeks"
    onset_order: Optional[int] = None    # e.g., 1 for first symptom, 2 for second


class ChatRequest(BaseModel):
    messages: List[Message]
    extracted_symptoms: Optional[List[str]] = []  # accumulate across turns

class ChatResponse(BaseModel):
    reply: str
    extracted_symptoms: List[str]
    top_conditions: List[dict]
    rag_sources: List[str]
    graph_followups: List[str]
    red_flags_detected: List[str]
    traversal_path: List[dict] = []

class GraphNode(BaseModel):
    id: str
    label: str
    type: str          # "symptom" | "condition" | "treatment"
    severity: Optional[str] = None
    description: Optional[str] = None

class GraphEdge(BaseModel):
    source: str
    target: str
    edge_type: str
    weight: float = 1.0

class GraphData(BaseModel):
    nodes: List[GraphNode]
    edges: List[GraphEdge]


# ---------------------------------------------------------------------------
# Core: build the enriched prompt
# ---------------------------------------------------------------------------

def build_system_prompt(
    extracted_symptoms: list,
    candidate_conditions: list,
    rag_context: str,
    followup_questions: list,
    red_flags: list,
) -> str:

    base = """You are SymptomAssist, a compassionate AI health assistant.
You have access to a medical knowledge graph and retrieved medical documents to inform your responses.
Your job is to guide the patient through a two-phase conversation:

PHASE 1 - DISCOVERY: Understand the symptoms, ask ONE targeted follow-up question.
PHASE 2 - CONFIRMATION: After 2-3 follow-ups, deliver a structured assessment.

ASSESSMENT FORMAT:
- Start with: "Based on what you've described..."
- State the most likely condition in plain language
- Explain what the condition typically involves
- Suggest appropriate home care steps
- Always end with: "Please consult a doctor for a proper diagnosis."
- If red flags are present, start with: "URGENT: [reason] — please seek emergency care immediately."

RULES:
- Be warm, clear, and concise (2-4 sentences per turn)
- Use "this may suggest" or "this sounds like it could be" — never claim to diagnose
- Ask only ONE follow-up question at a time
- Never recommend prescription drugs by name
- Ground your response in the retrieved medical context below
"""

    if red_flags:
        base += f"\n⚠️ RED FLAG SYMPTOMS DETECTED: {', '.join(red_flags)}\nIf these are present, immediately advise emergency care regardless of other context.\n"

    if extracted_symptoms:
        base += f"\nSYMPTOMS IDENTIFIED FROM PATIENT'S TEXT:\n{', '.join(extracted_symptoms)}\n"

    if candidate_conditions:
        base += "\nKNOWLEDGE GRAPH — BFS TRAVERSAL TOP CANDIDATE CONDITIONS:\n"
        for i, c in enumerate(candidate_conditions[:3], 1):
            base += f"  {i}. {c['display']} (traversal score: {c['score']}, severity: {c['severity']})\n"
            base += f"     Description: {c['description']}\n"

    if followup_questions:
        base += f"\nSUGGESTED FOLLOW-UP QUESTIONS (from knowledge graph — ask the most relevant one):\n"
        for q in followup_questions[:4]:
            base += f"  - Do you have {q}?\n"

    if rag_context:
        base += f"\nRETRIEVED MEDICAL CONTEXT (use this to ground your response):\n{rag_context}\n"

    base += "\nAlways communicate your assessment as a possibility, never a certainty."
    return base


# ---------------------------------------------------------------------------
# Chat endpoint
# ---------------------------------------------------------------------------

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        if not request.messages:
            raise HTTPException(status_code=400, detail="No messages provided")

        # Get the latest user message
        latest_user_msg = next(
            (m.content for m in reversed(request.messages) if m.role == "user"),
            ""
        )
        # --- Step 1: NLP extraction ---
        extraction = NLP.extract(latest_user_msg)
        all_symptoms = list(set(
            (request.extracted_symptoms or []) + extraction.symptoms
        ))

        # --- Step 2: Red flag check ---
        red_flags = check_red_flags(GRAPH, all_symptoms + (extraction.symptoms if extraction else []))

        # --- Step 3: BFS graph traversal ---
        candidates = traverse_graph(GRAPH, all_symptoms)
        followup_questions = []
        top_condition = None
        if candidates:
            top_condition = candidates[0]["condition_id"]
            followup_questions = get_followup_questions(GRAPH, top_condition)

        # --- Step 4: RAG retrieval ---
        rag_context = RAG.retrieve_context(latest_user_msg, top_k=2)
        rag_sources = [
            doc["title"]
            for doc in RAG.retrieve_raw(latest_user_msg, top_k=2)
        ]

        # --- Step 5: Build enriched system prompt ---
        system_prompt = build_system_prompt(
            extracted_symptoms=all_symptoms,
            candidate_conditions=candidates,
            rag_context=rag_context,
            followup_questions=followup_questions,
            red_flags=red_flags,
        )

        # --- Step 6: Call Groq with full context ---
        # Map roles to Groq roles ("user" -> "user", "model" -> "assistant")
        messages = [{"role": "system", "content": system_prompt}]
        for m in request.messages:
            role = "user" if m.role == "user" else "assistant"
            messages.append({"role": role, "content": m.content})

        try:
            chat_completion = GROQ.chat.completions.create(
                model="llama-3.1-8b-instant", 
                messages=messages,
                max_tokens=1000,
                temperature=0.3,
            )
            reply = chat_completion.choices[0].message.content
        except Exception as e:
            print(f"DEBUG: Groq API Error Detected: {e}")
            if "429" in str(e) or "limit" in str(e).lower():
                reply = "I'm sorry, I'm receiving too many requests from this account right now. Please try again soon."
            else:
                reply = f"I'm having trouble connecting to my reasoning engine. Error: {type(e).__name__}"

        return ChatResponse(
            reply=reply,
            extracted_symptoms=all_symptoms,
            top_conditions=[
                {
                    "display":       c["display"],
                    "score":         c["score"],
                    "severity":      c["severity"],
                    "condition_id":  c["condition_id"],
                    "traversal_path": c.get("traversal_path", []),
                }
                for c in candidates[:3]
            ],
            rag_sources=rag_sources,
            graph_followups=followup_questions[:4],
            red_flags_detected=red_flags,
            traversal_path=candidates[0].get("traversal_path", []) if candidates else [],
        )
    except Exception as overall_e:
        import traceback
        err_msg = traceback.format_exc()
        print("CRITICAL ERROR IN /chat ENDPOINT:")
        print(err_msg)
        with open("error_log.txt", "w") as f:
            f.write(err_msg)
        raise HTTPException(status_code=500, detail=str(overall_e))


# ---------------------------------------------------------------------------
# Debug endpoints
# ---------------------------------------------------------------------------

@app.post("/debug/analyse")
async def debug_analyse(body: dict):
    text = body.get("text", "")
    extraction = NLP.extract(text)
    candidates = traverse_graph(GRAPH, extraction.symptoms)
    rag_docs   = RAG.retrieve_raw(text, top_k=3)
    red_flags  = check_red_flags(GRAPH, extraction.symptoms)

    return {
        "input":                  text,
        "nlp_extracted_symptoms": extraction.symptoms,
        "nlp_negated_symptoms":   extraction.negated,
        "graph_candidates": [
            {
                "condition":  c["display"],
                "score":      c["score"],
                "severity":   c["severity"],
                "followups":  get_followup_questions(GRAPH, c["condition_id"])[:3],
            }
            for c in candidates[:3]
        ],
        "rag_retrieved": [
            {"title": d["title"], "relevance": d["relevance_score"]}
            for d in rag_docs
        ],
        "red_flags": red_flags,
    }


@app.post("/debug/traversal")
async def debug_traversal(body: dict):
    """
    Returns the full BFS traversal path for a given list of symptoms.
    Great for visualising how the graph inference works.
    """
    symptoms = body.get("symptoms", [])
    if isinstance(symptoms, str):
        symptoms = [s.strip() for s in symptoms.split(",") if s.strip()]

   
    candidates = traverse_graph(GRAPH, all_symptoms)

#  Apply temporal ranking
if candidates:
    candidates = rank_conditions_with_temporal_context(
        candidates,
        GRAPH,
        request.messages   #  IMPORTANT: pass full messages
    )
    traversal_path = candidates[0]["traversal_path"] if candidates else []

    return {
        "input_symptoms":   symptoms,
        "bfs_steps":        traversal_path,
        "steps_count":      len(traversal_path),
        "top_conditions":   [
            {"condition": c["display"], "score": c["score"], "severity": c["severity"]}
            for c in candidates[:5]
        ],
        "graph_stats":      graph_summary(GRAPH),
    }


# ---------------------------------------------------------------------------
# Graph data endpoint
# ---------------------------------------------------------------------------

@app.get("/graph-data", response_model=GraphData)
async def get_graph_data():
    """
    Serialise the full knowledge graph as D3-friendly JSON.
    Treatment nodes are included but marked separately so the frontend
    can choose whether to show them.
    """
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    seen_nodes = set()

    for node_id, attrs in GRAPH.nodes(data=True):
        if node_id in seen_nodes:
            continue
        seen_nodes.add(node_id)
        ntype = attrs.get("node_type", "symptom")
        label = attrs.get("display", node_id.replace("_", " ").title())
        nodes.append(GraphNode(
            id=node_id,
            label=label,
            type=ntype,
            severity=attrs.get("severity"),
            description=attrs.get("description"),
        ))

    for src, dst, attrs in GRAPH.edges(data=True):
        edges.append(GraphEdge(
            source=src,
            target=dst,
            edge_type=attrs.get("edge_type", "RELATES"),
            weight=attrs.get("weight", 1.0),
        ))

    return GraphData(nodes=nodes, edges=edges)


# ---------------------------------------------------------------------------
# Serve frontend
# ---------------------------------------------------------------------------

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def index():
    return FileResponse("static/index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
