"""
rag_pipeline.py
---------------
RAG (Retrieval-Augmented Generation) pipeline for the symptom chatbot.

Medical documents are now loaded from data/medical_docs.csv instead of
being hardcoded. The semantic engine uses sentence-transformers for retrieval.

Flow:
  1. Load medical documents from CSV at startup
  2. Build semantic index using pre-trained embeddings
  3. At query time, embed the user's symptom description
  4. Retrieve top-k most relevant document chunks via cosine similarity
  5. Return chunks as context to inject into the LLM prompt
"""

import csv
import math
import os
import re
from collections import defaultdict


# ---------------------------------------------------------------------------
# 1. CSV Loader
# ---------------------------------------------------------------------------

def load_documents_from_csv(csv_path: str) -> list[dict]:
    """
    Read medical_docs.csv and return a list of document dicts:
      {id, condition, title, content}
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Medical docs CSV not found: {csv_path}")

    documents: list[dict] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            documents.append({
                "id":        f"doc_{row['condition'].strip()}",
                "condition": row["condition"].strip(),
                "title":     row["title"].strip(),
                "content":   row["content"].strip(),
            })

    print(f"[RAG] Loaded {len(documents)} medical documents from CSV")
    return documents


# ---------------------------------------------------------------------------
# 2. Semantic vectoriser
# ---------------------------------------------------------------------------

class SemanticRetriever:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.documents = []
        self.doc_embeddings = None

    def index(self, documents: list[dict]):
        self.documents = documents
        texts = [d["content"] + " " + d["title"] for d in documents]
        if texts:
            self.doc_embeddings = self.model.encode(texts)
            print(f"[RAG] Indexed {len(documents)} documents using semantic embeddings")
        else:
            self.doc_embeddings = []

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        if not self.documents or len(self.doc_embeddings) == 0:
            return []

        import numpy as np
        from scipy.spatial.distance import cosine
        
        query_embedding = self.model.encode(query)
        
        scores = []
        for i, doc_vec in enumerate(self.doc_embeddings):
            # scipy.spatial.distance.cosine computes distance (1 - similarity)
            dist = cosine(query_embedding, doc_vec)
            score = 0.0 if np.isnan(dist) else 1.0 - dist
            scores.append((float(score), i))

        scores.sort(reverse=True)
        results = []
        for score, idx in scores[:top_k]:
            if score > 0:
                doc = self.documents[idx].copy()
                doc["relevance_score"] = round(score, 4)
                results.append(doc)
        return results


# ---------------------------------------------------------------------------
# 3. RAG Pipeline class
# ---------------------------------------------------------------------------

class RAGPipeline:
    def __init__(self, csv_path: str | None = None):
        resolved_path = csv_path or os.getenv("MEDICAL_DOCS_CSV_PATH", "data/medical_docs.csv")

        if resolved_path and os.path.exists(resolved_path):
            documents = load_documents_from_csv(resolved_path)
        else:
            # Fallback: empty (should not normally reach here)
            documents = []
            print(f"[RAG] WARNING: medical docs CSV not found at '{resolved_path}'; RAG context will be empty.")

        self.retriever = SemanticRetriever()
        self.retriever.index(documents)

    def retrieve_context(self, query: str, top_k: int = 3) -> str:
        """Return formatted context string for the LLM prompt."""
        docs = self.retriever.retrieve(query, top_k=top_k)
        if not docs:
            return ""
        parts = [f"[{doc['title']}]\n{doc['content']}" for doc in docs]
        return "\n\n---\n\n".join(parts)

    def retrieve_raw(self, query: str, top_k: int = 3) -> list[dict]:
        """Return raw document list with scores — useful for debugging."""
        return self.retriever.retrieve(query, top_k=top_k)


# ---------------------------------------------------------------------------
# 4. Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pathlib
    _here = pathlib.Path(__file__).parent.parent.parent
    csv_p = str(_here / "data" / "medical_docs.csv")

    rag = RAGPipeline(csv_path=csv_p)
    for q in [
        "I have a headache that throbs on one side with light sensitivity",
        "burning when I urinate and need to go frequently",
        "stomach cramps and vomiting after eating out",
    ]:
        print(f"\nQuery: '{q}'")
        for r in rag.retrieve_raw(q, top_k=2):
            print(f"  -> {r['title']} (score: {r['relevance_score']})")
