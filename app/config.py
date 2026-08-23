"""Application configuration and settings."""

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """App configuration loaded from environment variables."""

    # Flask
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key")
    DEBUG = os.getenv("FLASK_DEBUG", "True").lower() == "true"

    # Upload
    UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), "uploads")
    ALLOWED_EXTENSIONS = {"pdf", "txt"}
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50 MB

    # Chunking
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

    # Embedding model
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    # FAISS
    FAISS_INDEX_PATH = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", "faiss_index"
    )

    # Universal LLM API (Groq)
    LLM_API_KEY = os.getenv("LLM_API_KEY")
    LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
    LLM_MODEL = os.getenv("LLM_MODEL", "qwen/qwen3.6-27b")
