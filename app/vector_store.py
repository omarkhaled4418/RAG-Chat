"""FAISS vector store for indexing and similarity search."""

import json
import os
import re

import faiss
import numpy as np

from app.embedder import generate_embeddings, get_embedding_dim

_TOKEN_PATTERN = re.compile(r'[\w\u0600-\u06FF]+')
_DIGIT_PATTERN = re.compile(r'\d{4,}')
_SPACE_DASH_PATTERN = re.compile(r'[\s\-+]')


class VectorStore:
    """Manages a FAISS index along with chunk metadata."""

    def __init__(self, index_path: str, model_name: str = "all-MiniLM-L6-v2"):
        self.index_path = index_path
        self.model_name = model_name
        self.metadata_path = index_path + "_meta.json"

        self.index: faiss.IndexFlatL2 | None = None
        self.metadata: list[dict] = []  # parallel to index vectors

        self._load_or_create()

    # ── Persistence ──────────────────────────────────────────────────────

    def _load_or_create(self) -> None:
        """Load existing index from disk or create a new empty one."""
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            try:
                self.index = faiss.read_index(self.index_path)
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
                return
            except Exception:
                pass

        dim = get_embedding_dim(self.model_name)
        self.index = faiss.IndexFlatL2(dim)
        self.metadata = []

    def save(self) -> None:
        """Persist the index and metadata to disk."""
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        if self.index is not None:
            faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False)

    # ── Indexing ─────────────────────────────────────────────────────────

    def add_chunks(self, chunks: list[dict]) -> int:
        """
        Embed and add chunks to the index.

        Args:
            chunks: List of dicts with at least a 'text' key.

        Returns:
            Number of chunks added.
        """
        if not chunks:
            return 0

        texts = [c["text"] for c in chunks]
        embeddings = generate_embeddings(texts, self.model_name)

        self.index.add(embeddings)

        for chunk in chunks:
            self.metadata.append({
                "text": chunk["text"],
                "source": chunk.get("source", ""),
                "page": chunk.get("page", 0),
                "chunk_id": chunk.get("chunk_id", 0),
            })

        self.save()
        return len(chunks)

    # ── Search ───────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Search the index using fast hybrid search (exact keyword matching + semantic FAISS vector search).

        Args:
            query: Search query string.
            top_k: Number of results to return.

        Returns:
            List of metadata dicts with an added 'score' key.
        """
        if self.index is None or self.index.ntotal == 0 or not self.metadata:
            return []

        query_clean = query.strip().lower()
        exact_matches = []
        seen_indices = set()

        # 1. Exact string / keyword scan (vital for phone numbers, IDs, names, codes)
        tokens = [t.lower() for t in _TOKEN_PATTERN.findall(query_clean) if len(t) >= 3]
        digits = _DIGIT_PATTERN.findall(query_clean)

        for idx, meta in enumerate(self.metadata):
            text_lower = meta["text"].lower()
            score = 0
            if query_clean in text_lower:
                score += 10
            for d in digits:
                clean_text_digits = _SPACE_DASH_PATTERN.sub('', text_lower)
                if d in text_lower or d in clean_text_digits:
                    score += 20
            for t in tokens:
                if t in text_lower:
                    score += 2

            if score > 0:
                res = dict(meta)
                res["score"] = 0.01 / score
                exact_matches.append((score, idx, res))

        # Sort exact matches by highest score
        exact_matches.sort(key=lambda x: x[0], reverse=True)
        for _, idx, res in exact_matches[:top_k]:
            seen_indices.add(idx)

        # 2. Semantic vector search from FAISS
        query_embedding = generate_embeddings([query], self.model_name)
        k_search = min(top_k * 2, self.index.ntotal)
        distances, indices = self.index.search(query_embedding, k_search)

        semantic_matches = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx in seen_indices:
                continue
            res = dict(self.metadata[idx])
            res["score"] = float(dist)
            semantic_matches.append(res)
            seen_indices.add(idx)

        # Combine: prioritize exact matches first, followed by semantic matches
        combined = [res for _, _, res in exact_matches[:top_k]] + semantic_matches
        return combined[:top_k]

    # ── Management ───────────────────────────────────────────────────────

    def get_documents(self) -> list[dict]:
        """Return list of distinct indexed documents with statistics."""
        docs: dict[str, dict] = {}
        for meta in self.metadata:
            source = meta.get("source", "unknown")
            if source not in docs:
                docs[source] = {"source": source, "chunks": 0, "pages": set()}
            docs[source]["chunks"] += 1
            docs[source]["pages"].add(meta.get("page", 1))

        return [
            {
                "source": s,
                "chunks": data["chunks"],
                "page_count": len(data["pages"]),
            }
            for s, data in docs.items()
        ]

    def delete_document(self, source_name: str) -> dict:
        """Delete all chunks for a specific document source and rebuild index."""
        source_name = source_name.strip()
        remaining_meta = []
        deleted_count = 0

        for meta in self.metadata:
            if meta.get("source") == source_name:
                deleted_count += 1
            else:
                remaining_meta.append(meta)

        if deleted_count == 0:
            return {"deleted_chunks": 0, "status": "not_found"}

        dim = get_embedding_dim(self.model_name)
        new_index = faiss.IndexFlatL2(dim)

        if remaining_meta:
            texts = [m["text"] for m in remaining_meta]
            embeddings = generate_embeddings(texts, self.model_name)
            new_index.add(embeddings)

        self.index = new_index
        self.metadata = remaining_meta
        self.save()

        return {"deleted_chunks": deleted_count, "status": "deleted"}

    def clear(self) -> None:
        """Remove all vectors and metadata."""
        dim = get_embedding_dim(self.model_name)
        self.index = faiss.IndexFlatL2(dim)
        self.metadata = []
        self.save()

    @property
    def count(self) -> int:
        """Return the number of vectors in the index."""
        return self.index.ntotal if self.index else 0

