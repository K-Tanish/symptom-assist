"""
semantic_cache.py
-----------------
Local Semantic Caching layer for the POST /chat endpoint.

Reduces Groq LLM API calls by caching responses for semantically similar
queries within the same symptom context.  Uses cosine similarity on
sentence-transformer embeddings to detect near-duplicate queries.

Design
------
  - Cache key  = semantic embedding of user message
                 + SHA-256 hash of the sorted symptom-name list
  - On lookup, entries sharing the same symptom-context hash are
    scanned via cosine similarity of query embeddings.
  - If similarity >= threshold  →  cache HIT  (skip Groq call)
  - Entries expire after a configurable TTL.
  - Max cache size enforced via LRU eviction.

Why cache only the reply?
  NLP extraction, knowledge-graph traversal, and RAG retrieval are all
  local operations that execute in <50 ms.  The Groq LLM call is the
  sole source of network latency (~1-3 s) **and** the only consumer of
  external API quota.  Caching just the reply string keeps the cache
  small while delivering maximum impact.
"""

import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CacheEntry:
    """A single cached LLM response."""
    query_text: str
    query_embedding: np.ndarray
    context_hash: str          # SHA-256 prefix of sorted symptom names
    reply: str                 # the cached Groq reply text
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    hit_count: int = 0


# ---------------------------------------------------------------------------
# SemanticCache
# ---------------------------------------------------------------------------

class SemanticCache:
    """
    In-memory semantic cache for LLM responses.

    Groups entries by symptom-context hash, then uses cosine similarity
    on query embeddings to find matches within the same context group.
    """

    def __init__(
        self,
        model=None,
        similarity_threshold: float = 0.92,
        ttl_seconds: int = 3600,       # 1 hour default
        max_entries: int = 500,
    ):
        self._model = model            # SentenceTransformer instance (shared)
        self._threshold = similarity_threshold
        self._ttl = ttl_seconds
        self._max_entries = max_entries

        # context_hash  →  list[CacheEntry]
        self._store: dict[str, list[CacheEntry]] = {}
        self._total_entries: int = 0

        # Observability counters
        self._hits: int = 0
        self._misses: int = 0

        logging.info(
            "[SemanticCache] Initialized — threshold=%.2f, ttl=%ds, max=%d",
            similarity_threshold, ttl_seconds, max_entries,
        )

    # -- model property (lazy-loads if not injected) ------------------------

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
            logging.info("[SemanticCache] Loaded sentence-transformers model (fallback)")
        return self._model

    @model.setter
    def model(self, value):
        self._model = value

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _context_hash(symptom_names: list[str]) -> str:
        """Deterministic 16-char hash of the sorted, normalised symptom list."""
        normalised = sorted(s.strip().lower() for s in symptom_names if s.strip())
        raw = "|".join(normalised).encode()
        return hashlib.sha256(raw).hexdigest()[:16]

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two 1-D vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    # -- housekeeping -------------------------------------------------------

    def _purge_expired(self) -> None:
        """Drop entries older than TTL."""
        now = time.time()
        for ctx in list(self._store):
            before = len(self._store[ctx])
            self._store[ctx] = [
                e for e in self._store[ctx]
                if (now - e.created_at) < self._ttl
            ]
            removed = before - len(self._store[ctx])
            self._total_entries -= removed
            if not self._store[ctx]:
                del self._store[ctx]

    def _evict_lru(self) -> None:
        """Remove least-recently-accessed entries until under *max_entries*."""
        if self._total_entries <= self._max_entries:
            return

        # Flatten, sort oldest-access-first, drop excess
        flat = [
            (ctx, entry)
            for ctx, entries in self._store.items()
            for entry in entries
        ]
        flat.sort(key=lambda x: x[1].last_accessed)

        to_remove = self._total_entries - self._max_entries
        for i in range(min(to_remove, len(flat))):
            ctx, entry = flat[i]
            if ctx in self._store:
                self._store[ctx] = [e for e in self._store[ctx] if e is not entry]
                self._total_entries -= 1
                if not self._store[ctx]:
                    del self._store[ctx]

    # -- public API ---------------------------------------------------------

    def get(self, query_text: str, symptom_names: list[str]) -> Optional[str]:
        """
        Look up a cached reply for a semantically similar query
        within the same symptom context.

        Returns
        -------
        str or None
            The cached reply text, or ``None`` on a miss.
        """
        self._purge_expired()

        ctx = self._context_hash(symptom_names)
        entries = self._store.get(ctx)
        if not entries:
            self._misses += 1
            logging.debug("[SemanticCache] MISS (no context group) — %s", query_text[:80])
            return None

        query_emb = self.model.encode(query_text)

        best_score = 0.0
        best_entry: Optional[CacheEntry] = None
        for entry in entries:
            score = self._cosine_similarity(query_emb, entry.query_embedding)
            if score > best_score:
                best_score = score
                best_entry = entry

        if best_score >= self._threshold and best_entry is not None:
            best_entry.last_accessed = time.time()
            best_entry.hit_count += 1
            self._hits += 1
            logging.info(
                '[SemanticCache] HIT (%.4f) — "%s" ≈ "%s"',
                best_score, query_text[:50], best_entry.query_text[:50],
            )
            return best_entry.reply

        self._misses += 1
        logging.debug(
            "[SemanticCache] MISS (best=%.4f < %.2f) — %s",
            best_score, self._threshold, query_text[:80],
        )
        return None

    def put(self, query_text: str, symptom_names: list[str], reply: str) -> None:
        """
        Store an LLM reply in the cache.

        If a near-duplicate entry already exists (similarity ≥ 0.98),
        it is updated in-place instead of creating a duplicate.
        """
        ctx = self._context_hash(symptom_names)
        query_emb = self.model.encode(query_text)

        # --- deduplicate within the context group ---
        if ctx in self._store:
            for existing in self._store[ctx]:
                if self._cosine_similarity(query_emb, existing.query_embedding) >= 0.98:
                    existing.reply = reply
                    existing.last_accessed = time.time()
                    logging.debug("[SemanticCache] Updated existing entry — %s", query_text[:60])
                    return

        entry = CacheEntry(
            query_text=query_text,
            query_embedding=query_emb,
            context_hash=ctx,
            reply=reply,
        )

        self._store.setdefault(ctx, []).append(entry)
        self._total_entries += 1

        self._evict_lru()

        logging.info(
            '[SemanticCache] STORED — "%s" (ctx=%s, total=%d)',
            query_text[:60], ctx, self._total_entries,
        )

    def invalidate_context(self, symptom_names: list[str]) -> int:
        """Drop all entries for a given symptom context. Returns count removed."""
        ctx = self._context_hash(symptom_names)
        entries = self._store.pop(ctx, [])
        removed = len(entries)
        self._total_entries -= removed
        if removed:
            logging.info("[SemanticCache] Invalidated %d entries for context %s", removed, ctx)
        return removed

    def clear(self) -> None:
        """Wipe the entire cache."""
        self._store.clear()
        self._total_entries = 0
        logging.info("[SemanticCache] Cache cleared")

    def stats(self) -> dict:
        """Return cache performance statistics."""
        total = self._hits + self._misses
        return {
            "total_entries": self._total_entries,
            "context_groups": len(self._store),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / total, 4) if total > 0 else 0.0,
            "total_requests": total,
        }
