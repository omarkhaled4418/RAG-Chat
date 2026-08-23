"""RAG engine — orchestrates the retrieval-augmented generation pipeline."""

import os

from app.pdf_parser import extract_text_from_file
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

    def ingest_file(self, file_path: str) -> dict:
        """
        Process a document (.pdf or .txt): extract text → chunk → embed → store.

        Returns:
            {"filename": str, "pages": int, "chunks": int}
        """
        pages = extract_text_from_file(file_path)

        chunks = chunk_text(
            pages,
            chunk_size=self.config.CHUNK_SIZE,
            chunk_overlap=self.config.CHUNK_OVERLAP,
        )

        added = self.store.add_chunks(chunks)

        return {
            "filename": os.path.basename(file_path),
            "pages": len(pages),
            "chunks": added,
        }

    ingest_pdf = ingest_file

    def ingest_text(self, text: str, source_name: str = "raw_text.txt") -> dict:
        """
        Directly ingest raw text without saving a file to disk.

        Returns:
            {"filename": str, "pages": int, "chunks": int}
        """
        text = text.strip()
        if not text:
            return {"filename": source_name, "pages": 0, "chunks": 0}

        pages = [{"page": 1, "text": text, "source": source_name}]
        chunks = chunk_text(
            pages,
            chunk_size=self.config.CHUNK_SIZE,
            chunk_overlap=self.config.CHUNK_OVERLAP,
        )

        added = self.store.add_chunks(chunks)
        return {
            "filename": source_name,
            "pages": 1,
            "chunks": added,
        }

    # ── Search ───────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Direct vector & keyword search against indexed chunks."""
        return self.store.search(query, top_k=top_k)

    # ── Query ────────────────────────────────────────────────────────────

    def ask(
        self,
        question: str,
        session_id: str = "default",
        top_k: int = 5,
        system_prompt: str | None = None,
        temperature: float = 0.2,
    ) -> dict:
        """
        Answer a question using RAG.

        Returns:
            {"answer": str, "sources": list[dict]}
        """
        # Retrieve relevant chunks
        results = self.store.search(question, top_k=top_k)

        if not results:
            return {
                "answer": "I don't have any documents to search. Please upload a PDF, TXT file, or ingest text first.",
                "sources": [],
            }

        # Get conversation history
        history = get_history(session_id)

        # Generate answer
        answer = query_llm(
            query=question,
            context_chunks=results,
            history=history,
            model=self.config.LLM_MODEL,
            api_key=self.config.LLM_API_KEY,
            base_url=self.config.LLM_BASE_URL,
            system_prompt=system_prompt,
            temperature=temperature,
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

    def ask_stream(
        self,
        question: str,
        session_id: str = "default",
        top_k: int = 5,
        system_prompt: str | None = None,
        temperature: float = 0.2,
    ):
        """
        Stream an answer using RAG.

        Yields:
            dict with either {"token": str} or {"sources": list}
        """
        results = self.store.search(question, top_k=top_k)

        if not results:
            yield {"token": "I don't have any documents to search. Please upload a PDF, TXT file, or ingest text first."}
            yield {"sources": []}
            return

        history = get_history(session_id)

        full_answer = []
        for token in query_llm_stream(
            query=question,
            context_chunks=results,
            history=history,
            model=self.config.LLM_MODEL,
            api_key=self.config.LLM_API_KEY,
            base_url=self.config.LLM_BASE_URL,
            system_prompt=system_prompt,
            temperature=temperature,
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

    def get_documents(self) -> list[dict]:
        """Return list of indexed documents with chunk and page stats."""
        return self.store.get_documents()

    def delete_document(self, source_name: str) -> dict:
        """Delete all chunks for a specific document."""
        return self.store.delete_document(source_name)

    def clear_index(self) -> None:
        """Wipe the vector store."""
        self.store.clear()

    @property
    def document_count(self) -> int:
        """Number of chunks currently indexed."""
        return self.store.count

