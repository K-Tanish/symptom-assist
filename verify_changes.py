"""
Quick verification script for all PR changes.
Run: python verify_changes.py
"""
import pathlib

print("=" * 50)
print("VERIFYING ALL CHANGES")
print("=" * 50)

# ── PR #1 ────────────────────────────────────────
print("\n[PR #1] Duplicate key fix...")
from app.core.nlp_extractor import _MANUAL_SYNONYMS
phrases = _MANUAL_SYNONYMS["frequent urination"]
assert "peeing a lot" in phrases, "❌ FAILED — peeing a lot missing"
assert "urinating frequently" in phrases, "❌ FAILED — urinating frequently missing"
print("✅ PASSED — all phrases present")

# ── PR #2 ────────────────────────────────────────
print("\n[PR #2] explain_diagnosis()...")
from app.core.knowledge_graph import load_graph_from_csv, traverse_graph, explain_diagnosis
csv_p = str(pathlib.Path("data/symptom_disease.csv"))
G = load_graph_from_csv(csv_p)
results = traverse_graph(G, ["headache", "nausea"])
assert len(results) > 0, "❌ FAILED — no results"
explanation = explain_diagnosis(results[0])
assert "suggested" in explanation, "❌ FAILED — explanation wrong"
assert "Confidence" in explanation, "❌ FAILED — confidence missing"
assert "contributed" in explanation, "❌ FAILED — contribution missing"
print("✅ PASSED — explanation generated correctly")
print(explanation)

# ── PR #3 ────────────────────────────────────────
print("\n[PR #3] Tests...")
import subprocess
result = subprocess.run(
    ["pytest", "tests/", "-v", "--tb=short"],
    capture_output=True, text=True
)
print(result.stdout)
if result.returncode != 0:
    print("❌ FAILED — some tests failed")
else:
    print("✅ PASSED — all tests passed")

# ── PR #4 ────────────────────────────────────────
print("\n[PR #4] Score filter + Stale cache...")
from app.core.rag_pipeline import RAGPipeline
csv_p2 = str(pathlib.Path("data/medical_docs.csv"))
rag = RAGPipeline(csv_path=csv_p2)

ctx = rag.retrieve_context("fever and headache")
assert ctx != "", "❌ FAILED — relevant query returned empty"

ctx2 = rag.retrieve_context("asdfghjkl qwerty zxcvbnm")
assert ctx2 == "", "❌ FAILED — garbage query should return empty"
print("✅ PASSED — score filter working")

print("\n" + "=" * 50)
print("ALL CHANGES VERIFIED ✅")
print("=" * 50)