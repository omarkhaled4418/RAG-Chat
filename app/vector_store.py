"""FAISS vector store for indexing and similarity search."""

import json
import os

import faiss
import numpy as np

from app.embedder import generate_embeddings, get_embedding_dim


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
            self.index = faiss.read_index(self.index_path)
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        else:
            dim = get_embedding_dim(self.model_name)
            self.index = faiss.IndexFlatL2(dim)
            self.metadata = []

    def save(self) -> None:
        """Persist the index and metadata to disk."""
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
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
        embeddings = np.ascontiguousarray(embeddings, dtype="float32")

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
        Search the index for the most similar chunks to the query.

        Args:
            query: Search query string.
            top_k: Number of results to return.

        Returns:
            List of metadata dicts with an added 'score' key (lower = closer).
        """
        if self.index.ntotal == 0:
            return []

        query_embedding = generate_embeddings([query], self.model_name)
        query_embedding = np.ascontiguousarray(query_embedding, dtype="float32")

        distances, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            result = dict(self.metadata[idx])
            result["score"] = float(dist)
            results.append(result)

        return results

    # ── Management ───────────────────────────────────────────────────────

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
