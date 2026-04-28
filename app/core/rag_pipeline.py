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
                "title": (row.get("title") or "").strip(),
                "content": (row.get("content") or "").strip(),
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
        self.chunks = []
        self.chunk_embeddings = None

    def _chunk_text(self, text: str, chunk_size: int = 400, overlap: int = 50) -> list[str]:
        """
        Split text into chunks of roughly `chunk_size` words with `overlap` words.
        Using words as a proxy for tokens.
        """
        words = text.split()
        if not words:
            return []
        
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i : i + chunk_size])
            chunks.append(chunk)
            # If we've reached the end of the text, stop
            if i + chunk_size >= len(words):
                break
        return chunks

    def index(self, documents: list[dict], chunk_size: int = 400, overlap: int = 50):
        """
        Processes documents by splitting them into chunks and embedding each chunk.
        """
        self.chunks = []
        texts_to_embed = []

        for doc in documents:
            content = doc.get("content", "")
            title = doc.get("title", "")
            
            # Split the document into overlapping chunks
            doc_chunks = self._chunk_text(content, chunk_size, overlap)
            
            for i, chunk_text in enumerate(doc_chunks):
                # Store chunk with a reference to the original document
                chunk_data = {
                    "id": f"{doc['id']}_chunk_{i}",
                    "original_id": doc["id"],
                    "condition": doc["condition"],
                    "title": title,
                    "content": chunk_text
                }
                self.chunks.append(chunk_data)
                # Embed the chunk content along with the title for context
                texts_to_embed.append(f"{title}: {chunk_text}")

        if texts_to_embed:
            self.chunk_embeddings = self.model.encode(texts_to_embed)
            print(f"[RAG] Indexed {len(self.chunks)} chunks from {len(documents)} documents")
        else:
            self.chunk_embeddings = []

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        if not self.chunks or self.chunk_embeddings is None or len(self.chunk_embeddings) == 0:
            return []

        import numpy as np
        from scipy.spatial.distance import cosine
        
        query_embedding = self.model.encode(query)
        
        scores = []
        for i, chunk_vec in enumerate(self.chunk_embeddings):
            dist = cosine(query_embedding, chunk_vec)
            score = 0.0 if np.isnan(dist) else 1.0 - dist
            scores.append((float(score), i))

        scores.sort(reverse=True)
        results = []
        # Track original IDs to avoid returning multiple chunks from the same document
        # if they are highly similar (optional, but usually cleaner)
        seen_docs = set()
        
        for score, idx in scores:
            if len(results) >= top_k:
                break
            if score > 0:
                chunk = self.chunks[idx].copy()
                doc_id = chunk["original_id"]
                
                # Deduplication logic: only return the best chunk per document
                if doc_id not in seen_docs:
                    chunk["relevance_score"] = round(score, 4)
                    results.append(chunk)
                    seen_docs.add(doc_id)
                    
        return results


# ---------------------------------------------------------------------------
# 3. RAG Pipeline class
# ---------------------------------------------------------------------------

class RAGPipeline:
    def __init__(self, csv_path: str | None = None):
        if csv_path and os.path.exists(csv_path):
            documents = load_documents_from_csv(csv_path)
        else:
            # Fallback: empty (should not normally reach here)
            documents = []
            print("[RAG] WARNING: no medical_docs.csv found; RAG context will be empty.")

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
