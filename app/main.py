"""
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
import uuid
import pathlib
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
from groq import Groq
from dotenv import load_dotenv
import logging

from .core.error_handler import APIErrorHandler, retry_with_backoff
from .logging_config import setup_logging

import sys
import pathlib

_PROJECT_ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from app.core.knowledge_graph import (
    load_graph_from_csv, traverse_graph, find_candidate_conditions,
    get_followup_questions, get_treatment, check_red_flags, graph_summary
)
from app.core.rag_pipeline import RAGPipeline
from app.core.nlp_extractor import SymptomExtractor

# ---------------------------------------------------------------------------
# Resolve project paths and load environment
# ---------------------------------------------------------------------------
load_dotenv(_PROJECT_ROOT / ".env", override=True)


def get_groq_api_key() -> str:
    raw = os.getenv("GROQ_API_KEY", "")
    cleaned = raw.strip().strip('"').strip("'")

    print("👉 API KEY FROM ENV:", cleaned)   # ✅ ADD THIS LINE

    return cleaned


def ask_groq(prompt: str) -> str:
    api_key = get_groq_api_key()

    if not api_key:
        return "AI is running in limited mode: GROQ_API_KEY is missing."

    try:
        print("API KEY LOADED:", api_key[:10])

        client = Groq(api_key=api_key)

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a helpful medical assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.5,
        )

        return response.choices[0].message.content

    except Exception as e:
        err = str(e).lower()
        print("GROQ ERROR:", err)

        if "api" in err and "key" in err:
            return "❌ Invalid API Key. Please generate a NEW key and restart server."

        if "model" in err:
            return "❌ Model issue. Try changing model."

        return "❌ AI error: " + str(e)
# ---------------------------------------------------------------------------
# Resolve dataset paths (relative to the project root)
# ---------------------------------------------------------------------------
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
if get_groq_api_key():
    print("[startup] Groq client ready.")
else:
    print("[startup] GROQ_API_KEY missing. Running in fallback mode without LLM.")

# ---------------------------------------------------------------------------
# Server-side session store  { session_id -> {symptoms, last_active} }
# Sessions expire after 2 hours of inactivity.
# ---------------------------------------------------------------------------
SESSION_STORE: dict[str, dict] = {}
SESSION_TTL = timedelta(hours=2)


def _get_or_create_session(session_id: str | None) -> tuple[str, list[str]]:
    """Return (session_id, current_symptoms). Creates a new session if needed."""
    _purge_expired_sessions()
    if session_id and session_id in SESSION_STORE:
        SESSION_STORE[session_id]["last_active"] = datetime.utcnow()
        return session_id, SESSION_STORE[session_id]["symptoms"]
    new_id = str(uuid.uuid4())
    SESSION_STORE[new_id] = {"symptoms": [], "last_active": datetime.utcnow()}
    return new_id, []


def _purge_expired_sessions() -> None:
    """Drop sessions that have been inactive longer than SESSION_TTL."""
    cutoff = datetime.utcnow() - SESSION_TTL
    expired = [sid for sid, s in SESSION_STORE.items() if s["last_active"] < cutoff]
    for sid in expired:
        del SESSION_STORE[sid]

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

class TaggedSymptom(BaseModel):
    symptom: str
    status: str

class ChatRequest(BaseModel):
    messages: List[Message]
    extracted_symptoms: Optional[List[str]] = []  # accumulate across turns
    extracted_symptoms_detailed: Optional[List[TaggedSymptom]] = []

class ChatResponse(BaseModel):
    reply: str
    session_id: str                         # client must echo this on the next turn
    extracted_symptoms: List[str]
    extracted_symptoms_detailed: List[TaggedSymptom] = []
    symptom_timeline: List[str] = []
    top_conditions: List[dict]
    rag_sources: List[str]
    graph_followups: List[str]
    red_flags_detected: List[str]
    traversal_path: List[dict] = []
    journey_edges: List[dict] = []

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
    extracted_symptoms_detailed: list,
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

    if extracted_symptoms_detailed:
        base += "\nSYMPTOMS IDENTIFIED FROM PATIENT'S TEXT WITH STATUS:\n"
        for t in extracted_symptoms_detailed:
            if isinstance(t, dict):
                base += f"- {t['symptom']} (Status: {t['status']})\n"
            else:
                base += f"- {t.symptom} (Status: {t.status})\n"

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

FINAL_LINK_THRESHOLD = 0.65


@retry_with_backoff(max_retries=2, base_delay=1.0)
def call_groq_api(messages: list, model: str = "llama-3.1-8b-instant") -> str:
    """
    Call Groq API with proper error handling and retry logic.
    
    Args:
        messages: List of message dicts with role and content
        model: Model name to use
    
    Returns:
        str: API response content
    
    Raises:
        Various exceptions with user-friendly handling
    """
    chat_completion = GROQ.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=1000,
        temperature=0.3,
    )
    return chat_completion.choices[0].message.content


def merge_symptom_timeline(existing: List[str], newly_extracted: List[str]) -> List[str]:
    """Preserve first-seen order across turns while removing duplicates."""
    merged: List[str] = []
    seen = set()

    for symptom in (existing or []) + (newly_extracted or []):
        normalised = (symptom or "").strip().lower()
        if not normalised or normalised in seen:
            continue
        seen.add(normalised)
        merged.append(normalised)
    return merged


def build_journey_edges(symptom_timeline: List[str], candidates: List[dict]) -> List[dict]:
    """Build step-by-step symptom chain and threshold-gated first symptom->condition link."""
    edges: List[dict] = []

    for i in range(len(symptom_timeline) - 1):
        edges.append({
            "from": symptom_timeline[i],
            "to": symptom_timeline[i + 1],
            "edge_type": "SEQUENTIAL_SYMPTOM",
        })

    if symptom_timeline and candidates:
        top = candidates[0]
        top_score = float(top.get("score", 0.0))
        top_condition_id = top.get("condition_id", "")
        if top_condition_id and top_score >= FINAL_LINK_THRESHOLD:
            edges.append({
                "from": symptom_timeline[0],
                "to": top_condition_id,
                "edge_type": "FIRST_SYMPTOM_TO_CONDITION",
                "score": round(top_score, 3),
            })

    return edges

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

        # --- Session: load server-held symptom timeline ---
        session_id, prior_symptoms = _get_or_create_session(request.session_id)

        # --- Step 1: NLP extraction ---
        extraction = NLP.extract(latest_user_msg)
        all_symptoms = merge_symptom_timeline(prior_symptoms, extraction.symptoms)

        # Persist merged timeline back to session store
        SESSION_STORE[session_id]["symptoms"] = all_symptoms

        existing_detailed = [t.dict() if not isinstance(t, dict) else t for t in (request.extracted_symptoms_detailed or [])]
        tagged_dict = {t["symptom"]: t["status"] for t in existing_detailed}
        for t in extraction.tagged:
            tagged_dict[t["symptom"]] = t["status"]
        all_symptoms_detailed = [{"symptom": k, "status": v} for k, v in tagged_dict.items()]

        # --- Step 2: Red flag check ---
        red_flags = check_red_flags(GRAPH, all_symptoms + (extraction.symptoms if extraction else []))

        # --- Step 3: BFS graph traversal ---
        candidates = traverse_graph(GRAPH, all_symptoms)
        followup_questions = []
        top_condition = None
        if candidates:
            top_condition = candidates[0]["condition_id"]
            followup_questions = get_followup_questions(GRAPH, top_condition)

        journey_edges = build_journey_edges(all_symptoms, candidates)

        # --- Step 4: RAG retrieval ---
        rag_context = RAG.retrieve_context(latest_user_msg, top_k=2)
        rag_sources = [
            doc["title"]
            for doc in RAG.retrieve_raw(latest_user_msg, top_k=2)
        ]

        # --- Step 5: Build enriched system prompt ---
        system_prompt = build_system_prompt(
            extracted_symptoms=all_symptoms,
            extracted_symptoms_detailed=all_symptoms_detailed,
            candidate_conditions=candidates,
            rag_context=rag_context,
            followup_questions=followup_questions,
            red_flags=red_flags,
        )

        # --- Step 6: Call Groq ---
        if not get_groq_api_key():
            symptom_text = ", ".join(all_symptoms[:5]) if all_symptoms else "no clear symptom keywords"
            if candidates:
                top = candidates[0]["display"]
                reply = (
                    f"I noted these symptoms from your message: {symptom_text}. "
                    f"Based on this limited analysis, one possible match is {top}. "
                    "This is not a final diagnosis. "
                    "I'm running in limited mode because GROQ_API_KEY is not configured, "
                    "so replies may feel brief."
                )
            else:
                reply = (
                    f"I noted these symptoms from your message: {symptom_text}. "
                    "I could not map them confidently to a likely condition yet. "
                    "I'm running in limited mode because GROQ_API_KEY is not configured, "
                    "so replies may feel brief."
                )
        else:
            full_prompt = (
                f"{system_prompt}\n\n"
                f"Conversation so far:\n"
                + "\n".join([f"{m.role}: {m.content}" for m in request.messages])
            )
            reply = ask_groq(full_prompt)

        return ChatResponse(
            reply=reply,
            session_id=session_id,
            extracted_symptoms=all_symptoms,
            extracted_symptoms_detailed=[TaggedSymptom(**t) for t in all_symptoms_detailed],
            symptom_timeline=all_symptoms,
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
            journey_edges=journey_edges,
        )
    except Exception as overall_e:
        import traceback
        err_msg = traceback.format_exc()
        print("CRITICAL ERROR IN /chat ENDPOINT:")
        print(err_msg)
        APIErrorHandler.log_error(overall_e, "Critical error in /chat endpoint")
        with open("error_log.txt", "w") as f:
            f.write(err_msg)
        # Fail soft so frontend can keep working and display a helpful reply.
        return ChatResponse(
            reply=(
                "I'm sorry, I hit an internal processing issue while analyzing your message. "
                "Please try rephrasing once."
            ),
            extracted_symptoms=request.extracted_symptoms or [],
            extracted_symptoms_detailed=request.extracted_symptoms_detailed or [],
            symptom_timeline=request.extracted_symptoms or [],
            top_conditions=[],
            rag_sources=[],
            graph_followups=[],
            red_flags_detected=[],
            traversal_path=[],
            journey_edges=[],
        )


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

    candidates = traverse_graph(GRAPH, symptoms)
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
