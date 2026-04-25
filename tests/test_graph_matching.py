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


def test_multiple_symptoms_all_matched(graph):
    """Regression: early-exit bug — all 3 symptoms must contribute to scoring."""
    results = traverse_graph(graph, ["fever", "sore throat", "body aches"])
    assert len(results) > 0
    # Multiple symptom contributions push raw_score above any single edge weight
    assert results[0]["raw_score"] > 1.0, (
        f"raw_score={results[0]['raw_score']} — only one symptom seems to have contributed"
    )


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
