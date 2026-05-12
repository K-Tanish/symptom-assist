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
import hashlib
import json


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
        for i, row in enumerate(reader):
            documents.append({
                "id":        f"doc_{row['condition'].strip()}_{i}",
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
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", db_path: str = None):
        from sentence_transformers import SentenceTransformer
        import chromadb
        import pathlib
        
        self.model = SentenceTransformer(model_name)
        
        if db_path is None:
            _here = pathlib.Path(__file__).parent.parent.parent
            db_path = str(_here / "data" / "chroma_db")
            
        os.makedirs(db_path, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        self.collection = self.chroma_client.get_or_create_collection(name="medical_docs")

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
        Uses ChromaDB for persistence and vector storage.
        """
        csv_hash = hashlib.md5(
            json.dumps([d["id"] for d in documents], sort_keys=True).encode()
        ).hexdigest()

        stored_hash = (self.collection.metadata or {}).get("csv_hash")

        if self.collection.count() > 0 and stored_hash == csv_hash:
            print(f"[RAG] ChromaDB up to date ({self.collection.count()} chunks). Skipping indexing.")
            return
        elif self.collection.count() > 0:
            print("[RAG] CSV changed — rebuilding index...")
            self.chroma_client.delete_collection("medical_docs")
            self.collection = self.chroma_client.get_or_create_collection(
                name="medical_docs",
                metadata={"csv_hash": csv_hash}
            )

        ids = []
        texts_to_embed = []
        metadatas = []
        seen_ids = set()

        for doc in documents:
            content = doc.get("content", "")
            title = doc.get("title", "")
            
            # Split the document into overlapping chunks
            doc_chunks = self._chunk_text(content, chunk_size, overlap)
            
            for i, chunk_text in enumerate(doc_chunks):
                chunk_id = f"{doc['id']}_chunk_{i}"
                
                # Deduplicate within this batch
                if chunk_id in seen_ids:
                    continue
                seen_ids.add(chunk_id)
                
                # We need to embed title + content for better semantic matching
                embed_text = f"{title}: {chunk_text}"
                
                ids.append(chunk_id)
                texts_to_embed.append(embed_text)
                
                metadatas.append({
                    "original_id": doc["id"],
                    "condition": doc["condition"],
                    "title": title,
                    "content": chunk_text
                })

        if texts_to_embed:
            print(f"[RAG] Generating embeddings for {len(texts_to_embed)} chunks...")
            embeddings = self.model.encode(texts_to_embed).tolist()
            
            print(f"[RAG] Inserting {len(texts_to_embed)} chunks into ChromaDB...")
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=texts_to_embed
            )
            print("[RAG] Indexing complete.")
        else:
            print("[RAG] No texts to index.")

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        if self.collection.count() == 0:
            return []

        # Encode the query
        query_embedding = self.model.encode(query).tolist()
        
        # Query ChromaDB (fetch more than top_k to account for deduplication)
        fetch_k = top_k * 3
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=fetch_k,
            include=["metadatas", "distances"]
        )
        
        if not results["ids"] or not results["ids"][0]:
            return []
            
        final_results = []
        seen_docs = set()
        
        # ChromaDB returns a list of lists since we can query multiple vectors at once
        ids = results["ids"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]
        
        for i in range(len(ids)):
            if len(final_results) >= top_k:
                break
                
            # Convert ChromaDB's distance to a similarity score (rough proxy)
            # Assuming distance is a metric where lower is better.
            dist = distances[i]
            score = max(0.0, 1.0 - (dist / 2.0))
            
            meta = metadatas[i]
            doc_id = meta["original_id"]
            
            # Deduplication
            if doc_id not in seen_docs:
                chunk_data = {
                    "id": ids[i],
                    "original_id": doc_id,
                    "condition": meta["condition"],
                    "title": meta["title"],
                    "content": meta["content"],
                    "relevance_score": round(score, 4)
                }
                final_results.append(chunk_data)
                seen_docs.add(doc_id)
                
        return final_results


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

    def retrieve_context(self, query: str, top_k: int = 3, min_score: float = 0.3) -> str:
        """
        Return formatted context string for the LLM prompt.
        
        Args:
            query: User's symptom description
            top_k: Maximum number of documents to retrieve
            min_score: Minimum relevance score threshold (0.0 to 1.0).
                      Documents below this score are excluded to prevent
                      low-quality context reaching the LLM.
        
        Returns:
            str: Formatted context string, or empty string if nothing relevant found.
        """
        docs = self.retriever.retrieve(query, top_k=top_k)
        if not docs:
            return ""
        
        # Filter out low relevance chunks
        docs = [doc for doc in docs if doc["relevance_score"] >= min_score]
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
