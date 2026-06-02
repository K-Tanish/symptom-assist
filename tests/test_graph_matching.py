"""
tests/test_graph_matching.py
-----------------------------
Regression tests for symptom node matching in traverse_graph().
"""

import pathlib
import pytest

from app.core.knowledge_graph import load_graph_from_csv, traverse_graph

_CSV = pathlib.Path(__file__).parent.parent / "data" / "symptom_disease.csv"


@pytest.fixture(scope="module")
def graph():
    if not _CSV.exists():
        pytest.skip("symptom_disease.csv not found")
    return load_graph_from_csv(str(_CSV))


def max_single_symptom_weight(graph, symptoms):
    """
    Return the strongest single SUGGESTS edge among the matched symptom nodes.
    This gives us a graph-derived baseline instead of a hardcoded threshold.
    """
    max_weight = 0.0

    for symptom in symptoms:
        s = symptom.lower().strip()
        if s in graph and graph.nodes[s].get("node_type") == "symptom":
            for _, _, edge_data in graph.out_edges(s, data=True):
                if edge_data.get("edge_type") == "SUGGESTS":
                    max_weight = max(max_weight, edge_data.get("weight", 0.0))

    return round(max_weight, 4)


def test_multiple_symptoms_all_matched(graph):
    """Regression: early-exit bug — all 3 symptoms must contribute to scoring."""
    symptoms = ["fever", "sore throat", "body aches"]
    results = traverse_graph(graph, symptoms)

    assert len(results) > 0

    strongest_single_edge = max_single_symptom_weight(graph, symptoms)

    # The top multi-symptom result should be stronger than any one single edge.
    assert results[0]["raw_score"] > strongest_single_edge, (
        f"raw_score={results[0]['raw_score']} strongest_single_edge={strongest_single_edge}"
    )
    assert len(results[0]["contribution"]) > 1, "Expected multiple symptoms to contribute"


def test_multi_symptom_returns_more_candidates_than_single(graph):
    results_multi  = traverse_graph(graph, ["fever", "sore throat", "body aches"])
    results_single = traverse_graph(graph, ["fever"])
    assert len(results_multi) >= len(results_single)


def test_single_symptom_still_works(graph):
    assert len(traverse_graph(graph, ["fever"])) > 0


def test_single_symptom_headache(graph):
    assert len(traverse_graph(graph, ["headache"])) > 0


def test_empty_symptoms_returns_empty(graph):
    assert traverse_graph(graph, []) == []


def test_unrecognised_symptom_returns_empty(graph):
    assert traverse_graph(graph, ["xyzzy_not_a_real_symptom"]) == []


def test_mixed_known_unknown_symptoms(graph):
    results = traverse_graph(graph, ["fever", "xyzzy_unknown"])
    assert len(results) > 0


def test_duplicate_symptom_no_score_inflation(graph):
    """Same symptom twice must not double-count its score."""
    once  = traverse_graph(graph, ["fever"])
    twice = traverse_graph(graph, ["fever", "fever"])
    if once and twice:
        assert once[0]["raw_score"] == twice[0]["raw_score"]
