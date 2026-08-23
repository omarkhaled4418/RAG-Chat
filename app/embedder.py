import os
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

# Lazy-loaded singleton
_model: SentenceTransformer | None = None


def _get_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """Load the embedding model (cached after first call)."""
    global _model
    if _model is None:
        try:
            # Try offline / local cache first for speed and SSL resilience
            os.environ["HF_HUB_OFFLINE"] = "1"
            _model = SentenceTransformer(model_name, local_files_only=True)
        except Exception:
            os.environ.pop("HF_HUB_OFFLINE", None)
            _model = SentenceTransformer(model_name)

        if hasattr(_model, "eval"):
            _model.eval()
    return _model


def generate_embeddings(
    texts: list[str],
    model_name: str = "all-MiniLM-L6-v2",
) -> np.ndarray:
    """
    Generate embeddings for a list of text strings with optimized inference.

    Args:
        texts: List of text strings to embed.
        model_name: HuggingFace model name for sentence-transformers.

    Returns:
        numpy array of shape (len(texts), embedding_dim).
    """
    if not texts:
        return np.empty((0, get_embedding_dim(model_name)), dtype="float32")

    model = _get_model(model_name)
    show_bar = len(texts) > 20

    with torch.inference_mode():
        embeddings = model.encode(
            texts,
            show_progress_bar=show_bar,
            convert_to_numpy=True,
            batch_size=64 if len(texts) > 64 else 32,
        )
    return np.ascontiguousarray(embeddings, dtype="float32")


def get_embedding_dim(model_name: str = "all-MiniLM-L6-v2") -> int:
    """Return the dimensionality of the embedding model."""
    model = _get_model(model_name)
    return model.get_sentence_embedding_dimension()

