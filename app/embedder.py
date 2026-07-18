"""Embedding generation using sentence-transformers."""

import numpy as np
from sentence_transformers import SentenceTransformer

# Lazy-loaded singleton
_model: SentenceTransformer | None = None


def _get_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """Load the embedding model (cached after first call)."""
    global _model
    if _model is None:
        _model = SentenceTransformer(model_name)
    return _model


def generate_embeddings(
    texts: list[str],
    model_name: str = "all-MiniLM-L6-v2",
) -> np.ndarray:
    """
    Generate embeddings for a list of text strings.

    Args:
        texts: List of text strings to embed.
        model_name: HuggingFace model name for sentence-transformers.

    Returns:
        numpy array of shape (len(texts), embedding_dim).
    """
    model = _get_model(model_name)
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return embeddings


def get_embedding_dim(model_name: str = "all-MiniLM-L6-v2") -> int:
    """Return the dimensionality of the embedding model."""
    model = _get_model(model_name)
    return model.get_sentence_embedding_dimension()
