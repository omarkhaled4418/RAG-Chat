"""RAG engine — orchestrates the retrieval-augmented generation pipeline."""

import os

from app.pdf_parser import extract_text_from_pdf
from app.chunker import chunk_text
from app.vector_store import VectorStore
from app.llm import query_llm, query_llm_stream
from app.chat_history import add_message, get_history


class RAGEngine:
    """Ties together document ingestion, retrieval, and generation."""

    def __init__(self, config):
        self.config = config
        self.store = VectorStore(
            index_path=config.FAISS_INDEX_PATH,
            model_name=config.EMBEDDING_MODEL,
        )

    # ── Document ingestion ───────────────────────────────────────────────

    def ingest_pdf(self, pdf_path: str) -> dict:
        """
        Process a PDF: extract text → chunk → embed → store.

        Returns:
            {"filename": str, "pages": int, "chunks": int}
        """
        pages = extract_text_from_pdf(pdf_path)

        chunks = chunk_text(
            pages,
            chunk_size=self.config.CHUNK_SIZE,
            chunk_overlap=self.config.CHUNK_OVERLAP,
        )

        added = self.store.add_chunks(chunks)

        return {
            "filename": os.path.basename(pdf_path),
            "pages": len(pages),
            "chunks": added,
        }

    # ── Query ────────────────────────────────────────────────────────────

    def ask(self, question: str, session_id: str = "default", top_k: int = 3) -> dict:
        """
        Answer a question using RAG.

        Returns:
            {"answer": str, "sources": list[dict]}
        """
        # Retrieve relevant chunks
        results = self.store.search(question, top_k=top_k)

        if not results:
            return {
                "answer": "I don't have any documents to search. Please upload a PDF first.",
                "sources": [],
            }

        # Get conversation history
        history = get_history(session_id)

        # Generate answer
        answer = query_llm(
            query=question,
            context_chunks=results,
            history=history,
            model=self.config.GROQ_MODEL,
            api_key=self.config.GROQ_API_KEY,
        )

        # Save to history
        add_message(session_id, "user", question)
        add_message(session_id, "assistant", answer)

        # Build source references
        sources = []
        seen = set()
        for r in results:
            key = (r["source"], r["page"])
            if key not in seen:
                seen.add(key)
                sources.append({"source": r["source"], "page": r["page"]})

        return {"answer": answer, "sources": sources}

    def ask_stream(self, question: str, session_id: str = "default", top_k: int = 3):
        """
        Stream an answer using RAG.

        Yields:
            dict with either {"token": str} or {"sources": list}
        """
        results = self.store.search(question, top_k=top_k)

        if not results:
            yield {"token": "I don't have any documents to search. Please upload a PDF first."}
            yield {"sources": []}
            return

        history = get_history(session_id)

        full_answer = []
        for token in query_llm_stream(
            query=question,
            context_chunks=results,
            history=history,
            model=self.config.GROQ_MODEL,
            api_key=self.config.GROQ_API_KEY,
        ):
            full_answer.append(token)
            yield {"token": token}

        # Save to history
        add_message(session_id, "user", question)
        add_message(session_id, "assistant", "".join(full_answer))

        # Build source references
        sources = []
        seen = set()
        for r in results:
            key = (r["source"], r["page"])
            if key not in seen:
                seen.add(key)
                sources.append({"source": r["source"], "page": r["page"]})

        yield {"sources": sources}

    # ── Management ───────────────────────────────────────────────────────

    def clear_index(self) -> None:
        """Wipe the vector store."""
        self.store.clear()

    @property
    def document_count(self) -> int:
        """Number of chunks currently indexed."""
        return self.store.count
