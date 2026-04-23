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

            # Collect symptoms (symptom_1 … symptom_17, skip empties and leaked severity)
            symptoms = []
            for i in range(1, 18):
                key = f"symptom_{i}"
                val = (row.get(key) or "").strip()
                if val and val.lower() not in _VALID_SEV:
                    symptoms.append(val.lower())
                    symptom_condition_count[val.lower()] += 1

            rows.append({
                "condition_id": condition_id,
                "display":      display,
                "description":  description,
                "severity":     severity,
                "red_flags":    red_flags,
                "symptoms":     symptoms,
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

        for symptom in primary:
            if not G.has_node(symptom):
                G.add_node(symptom, node_type="symptom")
            # Rarer symptom → higher diagnostic weight
            freq   = symptom_condition_count.get(symptom, 1)
            weight = round(1.0 / freq, 4)
            G.add_edge(symptom, condition_id,
                       edge_type="SUGGESTS",
                       weight=weight)
        expected_onset_order=1,   # primary symptoms usually appear earlier
        typical_duration=3       # default (you can improve later)
    )
        for symptom in confirming:
            if not G.has_node(symptom):
                G.add_node(symptom, node_type="symptom")
            G.add_edge(condition_id, symptom,
                       edge_type="CONFIRMED_BY",
                       weight=0.9
                       expected_onset_order=2,
    typical_duration=5
    )

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

def traverse_graph(G: nx.DiGraph, symptoms: list[str]) -> list[dict]:
    """
    BFS traversal starting from every matched symptom node.

    Algorithm:
      1. For each user symptom, find matching graph symptom nodes
         (exact match or substring).
      2. Enqueue those symptom nodes.
      3. BFS: dequeue a node → visit all outgoing SUGGESTS edges
         → add weight to the target condition's accumulated score.
      4. Track visited nodes to avoid double-counting.
      5. Return conditions ranked by accumulated score, normalised to [0,1].

    Returns a list of dicts with keys:
      condition_id, display, description, severity, score, red_flags,
      traversal_path  (ordered list of symptom → condition steps taken)
    """
    if not symptoms:
        return []

    # -- Step 1: Match user symptoms to graph symptom nodes --
    matched_nodes: list[str] = []
    matched_set: set[str] = set()   # tracks already-added nodes for O(1) dedup
    unmatched: list[str] = []

    for user_sym in symptoms:
        u = user_sym.lower().strip()
        # Exact match
        if u in G and G.nodes[u].get("node_type") == "symptom":
            if u not in matched_set:
                matched_nodes.append(u)
                matched_set.add(u)
            continue
        # Substring match — iterate every node to collect ALL matches,
        # not just the first one.
        found = False
        for node in G.nodes:
            if G.nodes[node].get("node_type") == "symptom":
                if u in node or node in u:
                    found = True
                    if node not in matched_set:   # dedup check before append
                        matched_nodes.append(node)
                        matched_set.add(node)
        if not found:
            unmatched.append(u)


        matched_nodes = list(set(matched_nodes))

    if not matched_nodes:
        return []

    # -- Step 2 & 3: BFS traversal --
    condition_scores: dict[str, float] = defaultdict(float)
    traversal_path: list[dict] = []       # records each step taken
    visited: set[str] = set()

    queue: deque[str] = deque(matched_nodes)
    for n in matched_nodes:
        visited.add(n)

    while queue:
        current = queue.popleft()
        current_type = G.nodes[current].get("node_type", "")

        for _, neighbour, edge_data in G.out_edges(current, data=True):
            if edge_data.get("edge_type") != "SUGGESTS":
                continue
            if neighbour in visited:
                continue
            visited.add(neighbour)

            weight = edge_data.get("weight", 1.0)
            condition_scores[neighbour] += weight

            traversal_path.append({
                "from": current,
                "to":   neighbour,
                "weight": weight,
            })

            # Enqueue for further traversal (in case of multi-hop graphs)
            queue.append(neighbour)

    if not condition_scores:
        return []

    # -- Step 4: Normalise and rank --
    max_score = max(condition_scores.values()) or 1.0
    results = []
    for condition_id, score in sorted(condition_scores.items(), key=lambda x: -x[1]):
        node = G.nodes[condition_id]
        results.append({
            "condition_id":   condition_id,
            "display":        node.get("display", condition_id),
            "description":    node.get("description", ""),
            "severity":       node.get("severity", "low"),
            "score":          round(score / max_score, 3),
            "raw_score":      round(score, 4),
            "red_flags":      node.get("red_flags", []),
            "traversal_path": traversal_path,
        })

    return results[:7]   # top 7 candidates

# ---------------------------------------------------------------------------
# Temporal Ranking (NEW FEATURE)
# ---------------------------------------------------------------------------

def temporal_score(condition_id, G, messages):
    score = 0

    # Extract temporal info from messages
    for msg in messages:
    if msg.onset_order is None:
        continue

    for symptom_node, _, edge_data in G.in_edges(condition_id, data=True):
        if edge_data.get("edge_type") != "SUGGESTS":
            continue

        expected_order = edge_data.get("expected_onset_order")
        expected_duration = edge_data.get("typical_duration")

        # --- Onset order matching ---
        if expected_order is not None:
            if msg.onset_order == expected_order:
                score += 3
            elif msg.onset_order > expected_order:
                score += 1

        # --- Duration matching ---
        if msg.duration and expected_duration:
            try:
                user_days = int(msg.duration.split()[0])
                diff = abs(user_days - expected_duration)

                if diff <= 2:
                    score += 2
                elif diff <= 5:
                    score += 1
            except:
                pass


def rank_conditions_with_temporal_context(candidates, G, messages):
    for c in candidates:
        temp_score = temporal_score(c["condition_id"], G, messages)
        c["score"] += temp_score  # boost existing score

    return sorted(candidates, key=lambda x: x["score"], reverse=True)
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
