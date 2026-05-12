"""
tests/test_knowledge_graph.py
------------------------------
Unit tests for explain_diagnosis() in knowledge_graph.py
Uses a minimal in-memory graph — no CSV needed.
"""

import pytest
import networkx as nx
from app.core.knowledge_graph import (
    traverse_graph,
    get_followup_questions,
    explain_diagnosis,
)


@pytest.fixture
def simple_graph():
    """Minimal in-memory graph for testing without needing the CSV."""
    G = nx.DiGraph()

    # Symptom nodes
    G.add_node("headache", node_type="symptom")
    G.add_node("nausea", node_type="symptom")
    G.add_node("sensitivity to light", node_type="symptom")

    # Condition node
    G.add_node(
        "migraine",
        node_type="condition",
        display="Migraine",
        description="A headache disorder",
        severity="medium",
        red_flags=["sudden severe headache"],
    )

    # Edges
    G.add_edge("headache", "migraine",
               edge_type="SUGGESTS", weight=0.9, onset_index=1)
    G.add_edge("nausea", "migraine",
               edge_type="SUGGESTS", weight=0.7, onset_index=2)
    G.add_edge("migraine", "sensitivity to light",
               edge_type="CONFIRMED_BY", weight=0.9, onset_index=3)

    G.graph["red_flags_index"] = {"sudden severe headache"}
    return G


# ── explain_diagnosis() tests ────────────────────────────────────────────────

def test_explain_contains_condition_name(simple_graph):
    results = traverse_graph(simple_graph, ["headache", "nausea"])
    explanation = explain_diagnosis(results[0])
    assert "Migraine" in explanation


def test_explain_contains_confidence(simple_graph):
    results = traverse_graph(simple_graph, ["headache", "nausea"])
    explanation = explain_diagnosis(results[0])
    assert "Confidence" in explanation


def test_explain_contains_contribution(simple_graph):
    results = traverse_graph(simple_graph, ["headache", "nausea"])
    explanation = explain_diagnosis(results[0])
    assert "contributed" in explanation


def test_explain_contains_match_ratio(simple_graph):
    results = traverse_graph(simple_graph, ["headache", "nausea"])
    explanation = explain_diagnosis(results[0])
    assert "Matched" in explanation


def test_explain_shows_red_flag(simple_graph):
    results = traverse_graph(simple_graph, ["headache", "nausea"])
    explanation = explain_diagnosis(results[0])
    assert "sudden severe headache" in explanation


def test_explain_no_red_flag_when_none(simple_graph):
    # Remove red flags temporarily
    simple_graph.nodes["migraine"]["red_flags"] = []
    results = traverse_graph(simple_graph, ["headache", "nausea"])
    explanation = explain_diagnosis(results[0])
    assert "⚠️" not in explanation


def test_explain_single_symptom(simple_graph):
    """explain_diagnosis() should still work with just one symptom matched."""
    results = traverse_graph(simple_graph, ["headache"])
    assert len(results) > 0
    explanation = explain_diagnosis(results[0])
    assert "Migraine" in explanation


# ── get_followup_questions() tests ───────────────────────────────────────────

def test_followup_returns_confirmed_by_symptoms(simple_graph):
    questions = get_followup_questions(simple_graph, "migraine")
    assert "sensitivity to light" in questions


def test_followup_excludes_already_asked(simple_graph):
    questions = get_followup_questions(
        simple_graph, "migraine", asked_already=["sensitivity to light"]
    )
    assert "sensitivity to light" not in questions


def test_followup_empty_for_unknown_condition(simple_graph):
    questions = get_followup_questions(simple_graph, "unknown_condition")
    assert questions == []