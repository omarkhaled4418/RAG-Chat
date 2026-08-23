"""Flask application entry point and REST API routes."""

import json
import os
import time
import uuid

from flask import Blueprint, Response, jsonify, request
from werkzeug.utils import secure_filename

from app.chat_history import clear_all, clear_history, get_all_histories, get_history
from app.config import Config
from app.rag_engine import RAGEngine

api_bp = Blueprint("api", __name__)

# Lazy-initialized singleton engine
_engine: RAGEngine | None = None


def _get_engine() -> RAGEngine:
    global _engine
    if _engine is None:
        _engine = RAGEngine(Config)
    return _engine


def _allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in Config.ALLOWED_EXTENSIONS


# ── API Status, Root & Health Endpoints ─────────────────────────────────

@api_bp.route("/", methods=["GET", "POST"])
@api_bp.route("/api", methods=["GET"])
@api_bp.route("/health", methods=["GET"])
def index():
    if request.method == "POST":
        # Direct webhook / n8n convenience support at root POST /
        data = request.get_json(silent=True) or {}
        message = data.get("messages", data.get("message", ""))
        if isinstance(message, list):
            message = " ".join(str(m) for m in message)
        elif not isinstance(message, str):
            message = str(message)

        message = message.strip()
        if not message:
            return jsonify({"error": "Message is required"}), 400

        session_id = data.get("session_id", str(uuid.uuid4()))
        top_k = int(data.get("top_k", 5))
        system_prompt = data.get("system_prompt")
        temperature = float(data.get("temperature", 0.2))

        start_time = time.time()
        engine = _get_engine()
        result = engine.ask(
            message,
            session_id=session_id,
            top_k=top_k,
            system_prompt=system_prompt,
            temperature=temperature,
        )
        latency_ms = round((time.time() - start_time) * 1000, 2)

        result["session_id"] = session_id
        result["latency_ms"] = latency_ms
        return jsonify(result)

    # API Directory & Health Status
    engine = _get_engine()
    return jsonify({
        "status": "online",
        "service": "RAG Chat Headless REST API",
        "version": "2.0.0",
        "config": {
            "embedding_model": Config.EMBEDDING_MODEL,
            "llm_model": Config.LLM_MODEL,
            "llm_base_url": Config.LLM_BASE_URL,
            "chunk_size": Config.CHUNK_SIZE,
            "chunk_overlap": Config.CHUNK_OVERLAP,
        },
        "stats": {
            "chunks_indexed": engine.document_count,
            "documents_indexed": len(engine.get_documents()),
        },
        "endpoints": {
            "POST /api/chat": "Ask AI questions using document context (Body: {message, session_id?, top_k?, system_prompt?})",
            "POST /api/chat/stream": "Stream AI responses in real-time via Server-Sent Events (SSE)",
            "POST /api/search": "Fast similarity & keyword search without LLM (Body: {query, top_k?})",
            "POST /api/upload": "Upload .pdf or .txt files for indexing (multipart/form-data or raw binary)",
            "POST /api/documents/text": "Directly ingest raw text / markdown / notes (Body: {text, title?})",
            "GET /api/documents": "List all indexed documents, chunk counts, and page numbers",
            "DELETE /api/documents/<filename>": "Delete a document and remove all its vectors from memory",
            "GET /api/index/status": "Get current vector store statistics",
            "POST /api/index/clear": "Wipe all vectors and indexed documents from the system",
            "GET /api/history": "Retrieve chat history (Query: ?session_id=...)",
            "POST /api/history/clear": "Clear conversation history (Body: {session_id?})",
        },
    })


# ── Ingestion & Upload ───────────────────────────────────────────────────

@api_bp.route("/api/upload", methods=["POST"])
def upload():
    """Upload one or more PDF or TXT files for indexing (supports multipart & raw binary)."""
    results = []
    engine = _get_engine()

    # 1. Handle multipart/form-data uploads
    if request.files:
        files = []
        for key in request.files:
            files.extend(request.files.getlist(key))

        for file in files:
            if not file or not file.filename:
                continue
            if not _allowed_file(file.filename):
                allowed = ", ".join(f".{ext}" for ext in sorted(Config.ALLOWED_EXTENSIONS))
                results.append({"filename": file.filename, "error": f"Unsupported file type. Allowed: {allowed}"})
                continue

            filename = secure_filename(file.filename)
            save_path = os.path.join(Config.UPLOAD_FOLDER, filename)
            file.save(save_path)

            try:
                info = engine.ingest_file(save_path)
                results.append(info)
            except Exception as e:
                results.append({"filename": filename, "error": str(e)})

        return jsonify({"results": results, "total_chunks_indexed": engine.document_count})

    # 2. Handle raw binary data uploads
    raw_data = request.get_data()
    if raw_data:
        filename = (
            request.args.get("filename")
            or request.headers.get("X-Filename")
            or request.headers.get("X-File-Name")
            or "uploaded_file.txt"
        )
        filename = secure_filename(os.path.basename(filename))
        if not _allowed_file(filename):
            filename = f"{filename}.txt"

        save_path = os.path.join(Config.UPLOAD_FOLDER, filename)
        with open(save_path, "wb") as f:
            f.write(raw_data)

        try:
            info = engine.ingest_file(save_path)
            results.append(info)
        except Exception as e:
            results.append({"filename": filename, "error": str(e)})

        return jsonify({"results": results, "total_chunks_indexed": engine.document_count})

    return jsonify({"error": "No files or binary data provided"}), 400


@api_bp.route("/api/documents/text", methods=["POST"])
def ingest_raw_text():
    """
    Directly ingest raw text into vector store without file uploads.
    Body: {"text": "...", "title": "article.txt" (optional)}
    """
    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()
    title = data.get("title", f"text_snippet_{uuid.uuid4().hex[:6]}.txt").strip()

    if not text:
        return jsonify({"error": "Field 'text' is required and cannot be empty"}), 400

    engine = _get_engine()
    info = engine.ingest_text(text, source_name=title)
    return jsonify({
        "result": info,
        "total_chunks_indexed": engine.document_count,
    })


# ── Document Management ──────────────────────────────────────────────────

@api_bp.route("/api/documents", methods=["GET"])
def list_documents():
    """List all indexed documents and their chunk stats."""
    engine = _get_engine()
    documents = engine.get_documents()
    return jsonify({
        "documents": documents,
        "total_documents": len(documents),
        "total_chunks": engine.document_count,
    })


@api_bp.route("/api/documents/<path:filename>", methods=["DELETE"])
@api_bp.route("/api/documents/delete", methods=["POST", "DELETE"])
def delete_document(filename: str | None = None):
    """Delete a document and all its chunks from the vector store."""
    if not filename:
        data = request.get_json(silent=True) or {}
        filename = data.get("filename") or request.args.get("filename", "")

    filename = filename.strip()
    if not filename:
        return jsonify({"error": "Filename is required"}), 400

    engine = _get_engine()
    result = engine.delete_document(filename)
    if result.get("status") == "not_found":
        return jsonify({"error": f"Document '{filename}' not found in index", "result": result}), 404

    return jsonify({
        "status": "ok",
        "deleted_filename": filename,
        "result": result,
        "total_chunks_remaining": engine.document_count,
    })


# ── Search & Retrieval (Without LLM) ─────────────────────────────────────

@api_bp.route("/api/search", methods=["POST", "GET"])
def search():
    """
    Direct semantic & keyword search.
    Body/Query: {"query": "...", "top_k": 5}
    """
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        query = data.get("query", "").strip()
        top_k = int(data.get("top_k", 5))
    else:
        query = request.args.get("query", "").strip()
        top_k = int(request.args.get("top_k", 5))

    if not query:
        return jsonify({"error": "Query is required"}), 400

    start_time = time.time()
    engine = _get_engine()
    results = engine.search(query, top_k=top_k)
    latency_ms = round((time.time() - start_time) * 1000, 2)

    return jsonify({
        "query": query,
        "results": results,
        "count": len(results),
        "latency_ms": latency_ms,
    })


# ── Chat & Streaming ─────────────────────────────────────────────────────

@api_bp.route("/api/chat", methods=["POST"])
def chat():
    """
    POST /api/chat
    Body: {"message": "...", "session_id": "..." (opt), "top_k": 5 (opt), "system_prompt": "..." (opt), "temperature": 0.2 (opt)}
    """
    data = request.get_json(silent=True) or {}
    message = data.get("message", "").strip()

    if not message:
        return jsonify({"error": "Message is required"}), 400

    session_id = data.get("session_id", str(uuid.uuid4()))
    top_k = int(data.get("top_k", 5))
    system_prompt = data.get("system_prompt")
    temperature = float(data.get("temperature", 0.2))

    start_time = time.time()
    engine = _get_engine()
    result = engine.ask(
        message,
        session_id=session_id,
        top_k=top_k,
        system_prompt=system_prompt,
        temperature=temperature,
    )
    latency_ms = round((time.time() - start_time) * 1000, 2)

    result["session_id"] = session_id
    result["latency_ms"] = latency_ms

    return jsonify(result)


@api_bp.route("/api/chat/stream", methods=["POST"])
def chat_stream():
    """
    POST /api/chat/stream
    Streams the response as Server-Sent Events (SSE).
    """
    data = request.get_json(silent=True) or {}
    message = data.get("message", "").strip()

    if not message:
        return jsonify({"error": "Message is required"}), 400

    session_id = data.get("session_id", str(uuid.uuid4()))
    top_k = int(data.get("top_k", 5))
    system_prompt = data.get("system_prompt")
    temperature = float(data.get("temperature", 0.2))

    engine = _get_engine()

    def generate():
        for chunk in engine.ask_stream(
            message,
            session_id=session_id,
            top_k=top_k,
            system_prompt=system_prompt,
            temperature=temperature,
        ):
            yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ── History ──────────────────────────────────────────────────────────────

@api_bp.route("/api/history", methods=["GET"])
def get_chats_history():
    """Returns conversation history for a specific session or all sessions."""
    session_id = request.args.get("session_id")
    if session_id:
        history = get_history(session_id)
        return jsonify({"session_id": session_id, "history": history, "message_count": len(history)})

    all_histories = get_all_histories()
    return jsonify({
        "all_histories": all_histories,
        "total_sessions": len(all_histories),
    })


@api_bp.route("/api/history/clear", methods=["POST", "DELETE"])
@api_bp.route("/api/history", methods=["DELETE"])
def clear_session_history():
    """Clear conversation history for a specific session or all sessions."""
    data = request.get_json(silent=True) or {}
    session_id = data.get("session_id") or request.args.get("session_id")

    if session_id:
        clear_history(session_id)
        return jsonify({"status": "ok", "cleared_session": session_id})

    clear_all()
    return jsonify({"status": "ok", "cleared_all_sessions": True})


# ── Index Management ────────────────────────────────────────────────────

@api_bp.route("/api/index/status", methods=["GET"])
def index_status():
    """Return index stats and document counts."""
    engine = _get_engine()
    documents = engine.get_documents()
    return jsonify({
        "chunks_indexed": engine.document_count,
        "documents_count": len(documents),
        "embedding_model": Config.EMBEDDING_MODEL,
        "llm_model": Config.LLM_MODEL,
    })


@api_bp.route("/api/index/clear", methods=["POST", "DELETE"])
def clear_index():
    """Wipe all indexed vectors and metadata."""
    engine = _get_engine()
    engine.clear_index()
    return jsonify({
        "status": "ok",
        "message": "Vector store index cleared successfully",
        "chunks_indexed": engine.document_count,
    })

