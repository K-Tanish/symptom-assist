"""
knowledge_graph.py
------------------
Builds and queries a symptom → condition knowledge graph using NetworkX.
Data is loaded from data/symptom_disease.csv at startup.

Node types:
  - symptom   : e.g. "headache"
  - condition : e.g. "migraine"
  - treatment : (stored as node attribute; no separate treatment nodes)

Edge types:
  - symptom   -[SUGGESTS]->    condition  (weight reflects symptom rarity)
  - condition -[CONFIRMED_BY]-> symptom   (follow-up questions)

Graph Traversal:
  The diagnosis algorithm performs a BFS starting from each matched symptom
  node, visiting condition nodes reachable via SUGGESTS edges and accumulating
  path weights. This mirrors how a clinician reasons: starting from what the
  patient says, traversing the possibility space of conditions.
"""

import csv
import os
from collections import deque, defaultdict
import networkx as nx


# ---------------------------------------------------------------------------
# 1. CSV Loader — builds the graph from data/symptom_disease.csv
# ---------------------------------------------------------------------------

def load_graph_from_csv(csv_path: str) -> nx.DiGraph:
    """
    Read symptom_disease.csv and construct a weighted directed graph.

    CSV columns:
      condition, display, symptom_1 … symptom_17,
      severity, red_flags, description

    Edge weight for symptom → condition:
      Symptoms shared by fewer conditions get a *higher* weight
      (rarer = more diagnostic value). Computed as:
          weight = 1 / (number of conditions that share this symptom)
      Normalised to 0.3–1.0 range after all rows are processed.
    """
    G = nx.DiGraph()

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    rows = []
    symptom_condition_count: dict[str, int] = defaultdict(int)

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            condition_id  = (row.get("condition") or "").strip()
            display       = (row.get("display") or "").strip()
            description   = (row.get("description") or "").strip()
            severity      = (row.get("severity") or "low").strip()
            red_flag_str  = (row.get("red_flags") or "").strip()

            if not condition_id:   # skip blank rows
                continue

            # ── Self-healing column-shift detection ───────────────────────
            # Some rows have <17 symptoms, causing the named columns to shift.
            # Detect by checking if "severity" is not a valid level.
            _VALID_SEV = {"low", "medium", "high", ""}
            if severity.lower() not in _VALID_SEV:
                # Rotate: severity=red_flags, red_flag_str=description
                description  = red_flag_str
                red_flag_str = severity
                # The real severity may have been consumed by symptom_17
                _sev_cand = (row.get("symptom_17") or "").strip().lower()
                severity  = _sev_cand if _sev_cand in _VALID_SEV else "low"

            red_flags = [rf.strip() for rf in red_flag_str.split("|") if rf.strip()]
            # ─────────────────────────────────────────────────────────────

            # Collect symptoms (symptom_1 … symptom_17) with their order index
            symptoms_with_indices = []
            for i in range(1, 18):
                key = f"symptom_{i}"
                val = (row.get(key) or "").strip()
                if val and val.lower() not in _VALID_SEV:
                    symptoms_with_indices.append({
                        "name": val.lower(),
                        "index": i
                    })
                    symptom_condition_count[val.lower()] += 1

            rows.append({
                "condition_id": condition_id,
                "display":      display,
                "description":  description,
                "severity":     severity,
                "red_flags":    red_flags,
                "symptoms":     symptoms_with_indices,
            })

    # ----- Build graph -----
    for data in rows:
        condition_id = data["condition_id"]

        # Add condition node
        G.add_node(
            condition_id,
            node_type   = "condition",
            display     = data["display"],
            description = data["description"],
            severity    = data["severity"],
            red_flags   = data["red_flags"],
        )

        # Split symptoms: first half = primary, rest = confirming
        primary_count = max(1, len(data["symptoms"]) // 2)
        primary    = data["symptoms"][:primary_count]
        confirming = data["symptoms"][primary_count:]

        for item in primary:
            symptom = item["name"]
            index   = item["index"]
            if not G.has_node(symptom):
                G.add_node(symptom, node_type="symptom")
            # Rarer symptom → higher diagnostic weight
            freq   = symptom_condition_count.get(symptom, 1)
            weight = round(1.0 / freq, 4)
            G.add_edge(symptom, condition_id,
                       edge_type="SUGGESTS",
                       weight=weight,
                       onset_index=index) # Added temporal metadata

        for item in confirming:
            symptom = item["name"]
            index   = item["index"]
            if not G.has_node(symptom):
                G.add_node(symptom, node_type="symptom")
            G.add_edge(condition_id, symptom,
                       edge_type="CONFIRMED_BY",
                       weight=0.9,
                       onset_index=index)

    # Normalise SUGGESTS weights to [0.3, 1.0]
    suggests_weights = [
        d["weight"]
        for _, _, d in G.edges(data=True)
        if d.get("edge_type") == "SUGGESTS"
    ]
    if suggests_weights:
        w_min, w_max = min(suggests_weights), max(suggests_weights)
        span = w_max - w_min if w_max != w_min else 1.0
        for u, v, d in G.edges(data=True):
            if d.get("edge_type") == "SUGGESTS":
                normalised = 0.3 + 0.7 * (d["weight"] - w_min) / span
                G[u][v]["weight"] = round(normalised, 4)

    n_conditions = sum(1 for _, d in G.nodes(data=True) if d.get("node_type") == "condition")
    n_symptoms   = sum(1 for _, d in G.nodes(data=True) if d.get("node_type") == "symptom")
    print(f"[KG] Loaded {n_conditions} conditions, {n_symptoms} unique symptoms "
          f"| Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    return G


# ---------------------------------------------------------------------------
# 2. Graph Traversal — BFS from symptom nodes to condition nodes
# ---------------------------------------------------------------------------

def traverse_graph(G: nx.DiGraph, symptoms: list) -> list[dict]:
    """
    BFS traversal starting from every matched symptom node.
    Incorporates temporal context (onset order) into scoring.

    symptoms can be:
      - list[str]: old format (just names)
      - list[dict]: new format [{"name": "cough", "onset_order": 1}, ...]
    """
    if not symptoms:
        return []

    # Normalise input to dicts
    processed_symptoms = []
    for s in symptoms:
        if isinstance(s, str):
            processed_symptoms.append({"name": s, "onset_order": None})
        else:
            processed_symptoms.append(s)

    # -- Step 1: Match user symptoms to graph symptom nodes --
    matched_nodes: list[tuple[str, int | None]] = []
    matched_set: set[str] = set()   # tracks already-added nodes for O(1) dedup
    unmatched: list[str] = []

    for sym_obj in processed_symptoms:
        u = sym_obj["name"].lower().strip()
        # Exact match
        if u in G and G.nodes[u].get("node_type") == "symptom":
            if u not in matched_set:
                matched_nodes.append((u, sym_obj.get("onset_order")))
                matched_set.add(u)
            continue
        # Substring match
        found = False
        for node in G.nodes:
            if G.nodes[node].get("node_type") == "symptom":
                if u in node or node in u:
                    found = True
                    if node not in matched_set:
                        matched_nodes.append((node, sym_obj.get("onset_order")))
                        matched_set.add(node)
        if not found:
            unmatched.append(u)


    if not matched_nodes:
        return []

    # -- Step 2 & 3: Traverse matched symptom -> condition edges --
    condition_scores: dict[str, float] = defaultdict(float)
    symptom_contribution: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    traversal_path: list[dict] = []       # records each step taken
    visited_edges: set[tuple[str, str]] = set()

    for current, current_patient_order in matched_nodes:
        for _, neighbour, edge_data in G.out_edges(current, data=True):
            if edge_data.get("edge_type") != "SUGGESTS":
                continue
            if (current, neighbour) in visited_edges:
                continue
            visited_edges.add((current, neighbour))

            weight = edge_data.get("weight", 1.0)

            # --- TEMPORAL WEIGHTING LOGIC ---
            textbook_order = edge_data.get("onset_index", 99)
            temporal_multiplier = 1.0
            if current_patient_order is not None:
                if current_patient_order == textbook_order:
                    temporal_multiplier = 1.25  # Strong match
                elif abs(current_patient_order - textbook_order) <= 1:
                    temporal_multiplier = 1.1   # Close match

            final_weight = weight * temporal_multiplier
            condition_scores[neighbour] += final_weight
            symptom_contribution[neighbour][current] += final_weight

            traversal_path.append({
                "from": current,
                "to": neighbour,
                "weight": final_weight,
                "temporal_match": temporal_multiplier > 1.0,
            })

    if not condition_scores:
        return []

    # -- Step 4: Normalise and rank --
    max_score = max(condition_scores.values()) or 1.0
    results = []
    for condition_id, score in sorted(condition_scores.items(), key=lambda x: -x[1]):
        node = G.nodes[condition_id]
        matched_symptoms = list(symptom_contribution[condition_id].keys())
        match_count = len(matched_symptoms)

        # total possible symptoms for this condition
        total_symptoms = len([
            s for s in G.nodes
            if G.nodes[s].get("node_type") == "symptom" and G.has_edge(s, condition_id)
        ])

        # avoid division by zero
        ratio = match_count / total_symptoms if total_symptoms else 0

        # confidence level
        confidence = "low"
        if ratio > 0.6:
            confidence = "high"
        elif ratio > 0.3:
            confidence = "medium"
        results.append({
            "condition_id":   condition_id,
            "display":        node.get("display", condition_id),
            "description":    node.get("description", ""),
            "severity":       node.get("severity", "low"),
            "score":          round(score / max_score, 3),
            "raw_score":      round(score, 4),
            "red_flags":      node.get("red_flags", []),
            "traversal_path": traversal_path,
            "contribution": dict(symptom_contribution[condition_id]),
            "matched_symptoms": matched_symptoms,
            "total_symptoms": total_symptoms,
            "match_ratio": f"{match_count}/{total_symptoms}",
            "confidence": confidence,
        })

    return results[:7]   # top 7 candidates


# ---------------------------------------------------------------------------
# 3. Query helpers (same interface as before so main.py needs minimal changes)
# ---------------------------------------------------------------------------

def find_candidate_conditions(G: nx.DiGraph, symptoms: list[str]) -> list[dict]:
    """Alias kept for backwards-compatibility; delegates to traverse_graph."""
    return traverse_graph(G, symptoms)


def get_followup_questions(G: nx.DiGraph, condition_id: str,
                           asked_already: list[str] = None) -> list[str]:
    """Return CONFIRMED_BY symptom questions not yet asked."""
    asked_already = asked_already or []
    questions = []
    for _, symptom, data in G.out_edges(condition_id, data=True):
        if data.get("edge_type") == "CONFIRMED_BY":
            if symptom not in asked_already:
                questions.append(symptom)
    return questions


def get_treatment(G: nx.DiGraph, condition_id: str) -> list[str]:
    """
    Treatments are no longer stored as separate nodes.
    Return a generic care message referencing the description instead.
    """
    node = G.nodes.get(condition_id, {})
    desc = node.get("description", "")
    return [f"Refer to medical resources. {desc[:120]}..."] if desc else []


def check_red_flags(G: nx.DiGraph, symptoms: list[str]) -> list[str]:
    """Scan symptoms against red_flag lists for all condition nodes."""
    triggered: list[str] = []
    all_flags: set[str] = set()

    for node_id in G.nodes:
        if G.nodes[node_id].get("node_type") == "condition":
            for flag in G.nodes[node_id].get("red_flags", []):
                all_flags.add(flag.lower())

    for symptom in symptoms:
        s = symptom.lower()
        for flag in all_flags:
            if flag in s or s in flag:
                if flag not in triggered:
                    triggered.append(flag)

    return triggered


def graph_summary(G: nx.DiGraph) -> dict:
    symptom_nodes   = [n for n, d in G.nodes(data=True) if d.get("node_type") == "symptom"]
    condition_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "condition"]
    return {
        "total_nodes":    G.number_of_nodes(),
        "total_edges":    G.number_of_edges(),
        "symptoms":       len(symptom_nodes),
        "conditions":     len(condition_nodes),
        "condition_list": [G.nodes[c]["display"] for c in condition_nodes],
    }

def explain_diagnosis(result: dict) -> str:
    """
    Generates a human-readable explanation for a diagnosis result.
    Uses existing traversal_path and contribution data already 
    computed by traverse_graph().

    Args:
        result (dict): A single result dict from traverse_graph()

    Returns:
        str: Human-readable explanation of why this condition was suggested
    
    Example:
        >>> results = traverse_graph(G, ["headache", "nausea"])
        >>> print(explain_diagnosis(results[0]))
        Why 'Migraine' was suggested:
          - 'headache' contributed 56%
          - 'nausea' contributed 44%
        Confidence: MEDIUM
        Matched 2/4 known symptoms
    """
    lines = [f"Why '{result['display']}' was suggested:"]

    contribution = result.get("contribution", {})
    total = sum(contribution.values()) or 1

    for symptom, weight in sorted(contribution.items(), key=lambda x: -x[1]):
        percentage = round((weight / total) * 100)
        lines.append(f"  - '{symptom}' contributed {percentage}%")

    lines.append(f"\nConfidence: {result['confidence'].upper()}")
    lines.append(f"Matched {result['match_ratio']} known symptoms")

    if result.get("red_flags"):
        lines.append(f"\n⚠️ Red flags: {', '.join(result['red_flags'])}")

    return "\n".join(lines)

# ---------------------------------------------------------------------------
# 4. Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json, pathlib
    _here = pathlib.Path(__file__).parent.parent.parent
    csv_p = _here / "data" / "symptom_disease.csv"
    G = load_graph_from_csv(str(csv_p))
    print("Graph summary:", json.dumps(graph_summary(G), indent=2))

    print("\n--- BFS traversal: headache + nausea + sensitivity to light ---")
    results = traverse_graph(G, ["headache", "nausea", "sensitivity to light"])
    for r in results[:3]:
        print(f"  {r['display']} | score={r['score']} | severity={r['severity']}")
        print(f"  Follow-ups: {get_followup_questions(G, r['condition_id'])[:3]}")
