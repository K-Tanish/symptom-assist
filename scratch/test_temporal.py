
import sys
import os
import pathlib

# Add project root to path
_root = pathlib.Path(__file__).parent.parent
sys.path.append(str(_root))

from app.core.knowledge_graph import load_graph_from_csv, traverse_graph

SYMPTOM_CSV = str(_root / "data" / "symptom_disease.csv")
GRAPH = load_graph_from_csv(SYMPTOM_CSV)

def get_condition_score(results, cid):
    for r in results:
        if r['condition_id'] == cid:
            return r['raw_score']
    return 0.0

# Case 1: Fever first (1), then Cough (2)
case_1 = [
    {"name": "fever", "onset_order": 1},
    {"name": "cough", "onset_order": 2}
]

# Case 2: Cough first (1), then Fever (2)
case_2 = [
    {"name": "cough", "onset_order": 1},
    {"name": "fever", "onset_order": 2}
]

print("Testing Temporal Logic Differentiation...")
results_1 = traverse_graph(GRAPH, case_1)
results_2 = traverse_graph(GRAPH, case_2)

# Check Influenza (Textbook: Fever is 1st)
inf_1 = get_condition_score(results_1, 'influenza')
inf_2 = get_condition_score(results_2, 'influenza')

# Check Measles (Textbook: Fever 1st, Cough 2nd)
meas_1 = get_condition_score(results_1, 'measles')
meas_2 = get_condition_score(results_2, 'measles')

print(f"\n--- ANALYSIS: Influenza (Pattern: Fever then Cough) ---")
print(f"Score (Fever 1st, Cough 2nd): {inf_1}")
print(f"Score (Cough 1st, Fever 2nd): {inf_2}")
if inf_1 > inf_2:
    print("SUCCESS: Influenza scored higher when sequence matched its pattern.")

print(f"\n--- ANALYSIS: Measles (Pattern: Fever 1st, Cough 2nd) ---")
print(f"Score (Fever 1st, Cough 2nd): {meas_1}")
print(f"Score (Cough 1st, Fever 2nd): {meas_2}")
if meas_1 > meas_2:
    print("SUCCESS: Measles scored higher when sequence matched its pattern.")
