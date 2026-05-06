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
from pydantic import BaseModel, Field
from typing import List, Optional
from groq import AsyncGroq
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
load_dotenv()
setup_logging(log_dir="logs", level=logging.INFO)


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
print("[startup] Groq client ready.")
GROQ = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))

# ---------------------------------------------------------------------------
# Server-side session store: sessionId -> { symptoms: list[dict], last_active: datetime }
# Sessions expire after 2 hours of inactivity.
# ---------------------------------------------------------------------------
SESSION_STORE: dict[str, dict] = {}
SESSION_TTL = timedelta(hours=2)


def _get_or_create_session(session_id: str | None) -> tuple[str, list[dict]]:
    """Return (session_id, current_symptoms). Symptoms are now dicts with metadata."""
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
    role: str = Field(..., pattern="^(user|assistant|model)$")
    content: str = Field(..., min_length=1, max_length=1000)

class SymptomDetail(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    onset_order: Optional[int] = Field(None, ge=1, le=50)
    duration: Optional[str] = Field(None, max_length=100)
    severity: Optional[str] = Field(None, max_length=50)

class TaggedSymptom(BaseModel):
    symptom: str
    status: str

class ChatRequest(BaseModel):
    messages: List[Message]
    extracted_symptoms: Optional[List[str]] = []  # accumulate across turns
    extracted_symptoms_detailed: Optional[List[TaggedSymptom]] = []
    messages: List[Message] = Field(..., max_length=20)
    session_id: Optional[str] = Field(None, max_length=100)
    extracted_symptoms: Optional[List[str]] = Field([], max_length=30)
    temporal_context: Optional[List[SymptomDetail]] = Field([], max_length=30)

class ChatResponse(BaseModel):
    reply: str
    session_id: str                         # client must echo this on the next turn
    extracted_symptoms: List[str]
    extracted_symptoms_detailed: List[TaggedSymptom] = []
    symptom_timeline: List[str] = []
    temporal_context: List[SymptomDetail] = [] # New: returned to frontend
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

class SummaryResponse(BaseModel):
    text: str
    data: dict


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
    extracted_symptoms,
    candidate_conditions,
    rag_context,
    followup_questions,
    red_flags,
    has_noise=False,
)-> str:

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
- Never contradict yourself in the same response
- Do not say you are unsure if valid symptoms are already identified
- Be warm, clear, and concise (2-4 sentences per turn)
- Use "this may suggest" or "this sounds like it could be" — never claim to diagnose
- Ask only ONE follow-up question at a time
- Never recommend prescription drugs by name
- Ground your response in the retrieved medical context below
- Avoid repeating the same conclusion twice
- Do not restate the same condition multiple times
- Speak naturally like a doctor, not like a report
- If the user's input is unclear, interpret it intelligently instead of rejecting it
- Combine insights into one smooth explanation
"""
    if has_noise:
        base += "\nNOTE: The user's input may contain unclear or extra information. Focus on the valid symptoms and guide gently.\n"
        
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


def build_clinical_summary_text(
    symptoms: List[dict],
    candidates: List[dict],
    red_flags: List[str],
    rag_sources: List[str]
) -> str:
    """Build a sober, clinical-ready text block for clinicians."""
    lines = []
    lines.append("SYMPTOMASSIST - CLINICAL SUMMARY REPORT")
    lines.append("=" * 50)
    lines.append(f"Generated on: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    
    lines.append("\n[!] MEDICAL DISCLAIMER")
    lines.append("This document was generated by SymptomAssist, an AI-powered educational tool. It is NOT a medical record or diagnosis. It is intended to help a patient communicate symptoms to a healthcare professional. Do not use this to self-diagnose or delay seeking professional care.")
    
    if red_flags:
        lines.append("\n[!] URGENT: RED FLAG SYMPTOMS DETECTED")
        for rf in red_flags:
            lines.append(f"  - {rf.upper()}")
        lines.append("Recommended Action: Seek immediate medical attention or emergency care.")

    lines.append("\nREPORTED SYMPTOM TIMELINE")
    lines.append("-" * 30)
    # Sort by onset order
    sorted_symptoms = sorted(
        symptoms,
        key=lambda x: x.get("onset_order") if x.get("onset_order") is not None else 999
    )
    for s in sorted_symptoms:
        name = s['name'].replace("_", " ").title()
        onset = f" (Order: {s['onset_order']})" if s.get('onset_order') else ""
        dur = f" | Duration: {s['duration']}" if s.get('duration') else ""
        sev = f" | Severity: {s['severity']}" if s.get('severity') else ""
        lines.append(f"• {name}{onset}{dur}{sev}")

    lines.append("\nAI-GRAPH POSSIBILITY MATCHES")
    lines.append("-" * 30)
    lines.append("Matches found in knowledge graph (ordered by traversal score). These are NOT diagnoses.")
    for i, c in enumerate(candidates[:3], 1):
        conf = (c.get('confidence') or 'Low').upper()
        lines.append(f"{i}. {c['display']} (Confidence: {conf})")
        lines.append(f"   Note: {c['description']}")

    lines.append("\nEDUCATIONAL CONTEXT SOURCES")
    lines.append("-" * 30)
    if rag_sources:
        for src in rag_sources:
            lines.append(f"• {src}")
    else:
        lines.append("No specific educational documents were retrieved for this session.")

    lines.append("\n" + "=" * 50)
    lines.append("End of Summary")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Chat endpoint
# ---------------------------------------------------------------------------

FINAL_LINK_THRESHOLD = 0.65


@retry_with_backoff(max_retries=2, base_delay=1.0)
async def call_groq_api(messages: list, model: str = "llama-3.1-8b-instant") -> str:
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
    chat_completion = await GROQ.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=1000,
        temperature=0.3,
    )
    return chat_completion.choices[0].message.content


def merge_symptom_timeline(existing: List[dict], newly_extracted: List[str]) -> List[dict]:
    """Preserve first-seen order across turns while removing duplicates."""
    merged: List[dict] = [dict(s) for s in (existing or [])]
    seen = {s["name"].lower() for s in merged}

    for symptom in (newly_extracted or []):
        normalised = (symptom or "").strip().lower()
        if not normalised or normalised in seen:
            continue
        seen.add(normalised)
        # Default to the next available order index
        merged.append({
            "name": normalised,
            "onset_order": len(merged) + 1,
            "duration": None,
            "severity": None
        })
    return merged


def build_journey_edges(symptom_timeline: List[dict], candidates: List[dict]) -> List[dict]:
    """Build step-by-step symptom chain based on onset_order."""
    edges: List[dict] = []

    # Sort symptoms by their temporal onset for the journey visualization
    sorted_symptoms = sorted(
        symptom_timeline,
        key=lambda x: x.get("onset_order") if x.get("onset_order") is not None else 999
    )
    names = [s["name"] for s in sorted_symptoms]

    for i in range(len(names) - 1):
        edges.append({
            "from": names[i],
            "to": names[i + 1],
            "edge_type": "SEQUENTIAL_SYMPTOM",
        })

    if names and candidates:
        top = candidates[0]
        top_score = float(top.get("score", 0.0))
        top_condition_id = top.get("condition_id", "")
        if top_condition_id and top_score >= FINAL_LINK_THRESHOLD:
            edges.append({
                "from": names[0],
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

        # --- Step 0: Session Management ---
        # Retrieve existing symptoms and session ID (or create new ones)
        session_id, prior_symptoms = _get_or_create_session(request.session_id)

        # Get the latest user message
        latest_user_msg = next(
            (m.content for m in reversed(request.messages) if m.role == "user"),
            ""
        )

        # --- Step 1: NLP extraction ---
        extraction = NLP.extract(latest_user_msg)

        # 🚨 Handle mixed valid + invalid input
        noise_message = ""
        if extraction.symptoms and getattr(extraction, 'noise', None):
            noise_message = f"I understood {', '.join(extraction.symptoms)}, but some parts of your input were unclear."
        elif not extraction.symptoms:
            noise_message = "I couldn't identify any valid symptoms. Please describe your symptoms clearly."

        # Temporal Context Logic
        if extraction.symptoms:
            all_symptoms_data = merge_symptom_timeline(prior_symptoms, extraction.symptoms)
        else:
            all_symptoms_data = list(prior_symptoms)

        if request.temporal_context:
            for ctx in request.temporal_context:
                ctx_name = ctx.name.lower().strip()
                found = False
                for sym in all_symptoms_data:
                    if sym["name"] == ctx_name:
                        if ctx.onset_order is not None: sym["onset_order"] = ctx.onset_order
                        if ctx.duration: sym["duration"] = ctx.duration
                        if ctx.severity: sym["severity"] = ctx.severity
                        found = True
                        break
                if not found:
                    all_symptoms_data.append(ctx.dict())

        # Persist merged timeline back to session store
        SESSION_STORE[session_id]["symptoms"] = all_symptoms_data
        all_symptom_names = [s["name"] for s in all_symptoms_data]

        existing_detailed = [t.dict() if not isinstance(t, dict) else t for t in (request.extracted_symptoms_detailed or [])]
        tagged_dict = {t["symptom"]: t["status"] for t in existing_detailed}
        for t in extraction.tagged:
            tagged_dict[t["symptom"]] = t["status"]
        all_symptoms_detailed = [{"symptom": k, "status": v} for k, v in tagged_dict.items()]

        # --- Step 2: Red flag check ---
        red_flags = check_red_flags(GRAPH, all_symptom_names)

        # --- Step 3: BFS graph traversal ---
        candidates = traverse_graph(GRAPH, all_symptoms_data)
        followup_questions = []
        top_condition = [
            {
                "display": c["display"],
                "score": c["score"],
                "severity": c["severity"],
                "condition_id": c["condition_id"],
                "traversal_path": c.get("traversal_path", []),

                # 🔥 ADD THESE (your XAI fields)
                "confidence": c.get("confidence"),
                "match_ratio": c.get("match_ratio"),
                "matched_symptoms": c.get("matched_symptoms"),
                "contribution": c.get("contribution"),
            }
            for c in candidates[:3]
        ]
        top_condition_id = candidates[0]["condition_id"] if candidates else None
        followup_questions = get_followup_questions(GRAPH, top_condition_id) if top_condition_id else []

        journey_edges = build_journey_edges(all_symptoms_data, candidates)

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
            extracted_symptoms=all_symptom_names,
            candidate_conditions=candidates,
            rag_context=rag_context,
            followup_questions=followup_questions,
            red_flags=red_flags,
            has_noise=bool(extraction.noise),
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
        # --- Step 6: Call Groq with full context ---
        # Map roles to Groq roles ("user" -> "user", "model" -> "assistant")
        messages = [{"role": "system", "content": system_prompt}]
        for m in request.messages:
            role = "user" if m.role == "user" else "assistant"
            messages.append({"role": role, "content": m.content})

        try:
            reply = await call_groq_api(messages)
            if noise_message:
                reply = f"{noise_message}\n\n{reply}"
        except Exception as e:
            # Log full error for debugging
            APIErrorHandler.log_error(e, "Groq API call failed in /chat endpoint")
            # Get user-friendly message
            reply = APIErrorHandler.get_user_message(e)

        return ChatResponse(
            reply=reply,
            session_id=session_id,
            extracted_symptoms=all_symptom_names,
            symptom_timeline=all_symptom_names,
            temporal_context=[SymptomDetail(**s) for s in all_symptoms_data],
            top_conditions=top_condition,
            rag_sources=rag_sources,
            graph_followups=followup_questions[:4],
            red_flags_detected=red_flags,
            traversal_path=candidates[0].get("traversal_path", []) if candidates else [],
            journey_edges=journey_edges,
        ))
    except Exception as overall_e:
        import traceback
        err_msg = traceback.format_exc()
        print("CRITICAL ERROR IN /chat ENDPOINT:")
        print(err_msg)
        APIErrorHandler.log_error(overall_e, "Critical error in /chat endpoint")
        with open("error_log.txt", "w", encoding="utf-8") as f:
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
        raise HTTPException(status_code=500, detail=APIErrorHandler.get_user_message(overall_e)) from overall_e


@app.post("/session/clear")
async def clear_session(body: dict):
    """Clears the symptom timeline for a given session (used by 'New Chat')."""
    session_id = body.get("session_id")
    if session_id and session_id in SESSION_STORE:
        del SESSION_STORE[session_id]
    return {"cleared": True}


@app.get("/summary/{session_id}", response_model=SummaryResponse)
async def get_summary(session_id: str):
    """Generates a stable, clinical summary for the given session."""
    if session_id not in SESSION_STORE:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    
    session_data = SESSION_STORE[session_id]
    symptoms = session_data["symptoms"]
    
    if not symptoms:
        return SummaryResponse(
            text="No symptoms recorded yet. Please describe your symptoms in the chat first.",
            data={}
        )

    # Re-calculate matches and flags to ensure summary is up-to-date
    all_symptom_names = [s["name"] for s in symptoms]
    red_flags = check_red_flags(GRAPH, all_symptom_names)
    candidates = traverse_graph(GRAPH, symptoms)
    
    # RAG retrieval for summary context (using all symptoms as a combined query)
    combined_query = ", ".join(all_symptom_names)
    rag_raw = RAG.retrieve_raw(combined_query, top_k=3)
    rag_sources = [doc["title"] for doc in rag_raw]
    
    summary_text = build_clinical_summary_text(
        symptoms=symptoms,
        candidates=candidates,
        red_flags=red_flags,
        rag_sources=rag_sources
    )
    
    return SummaryResponse(
        text=summary_text,
        data={
            "symptoms": symptoms,
            "top_conditions": candidates[:3],
            "red_flags": red_flags,
            "rag_sources": rag_sources
        }
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
    uvicorn.run(app, host="127.0.0.1", port=8080)
    uvicorn.run(
        app,
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8000")),
    )
