"""Flask application entry point and REST API routes."""

import json
import os
import uuid

from flask import Blueprint, Response, current_app, jsonify, render_template, request
from werkzeug.utils import secure_filename

from app.chat_history import clear_history
from app.config import Config
from app.rag_engine import RAGEngine

api_bp = Blueprint("api", __name__)

# Lazy-initialized engine
_engine: RAGEngine | None = None


def _get_engine() -> RAGEngine:
    global _engine
    if _engine is None:
        _engine = RAGEngine(Config)
    return _engine


def _allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in Config.ALLOWED_EXTENSIONS


# ── Pages ────────────────────────────────────────────────────────────────

@api_bp.route("/")
def index():
    return render_template("index.html")


# ── Upload ───────────────────────────────────────────────────────────────

@api_bp.route("/api/upload", methods=["POST"])
def upload():
    """Upload one or more PDF files for indexing."""
    if "files" not in request.files:
        return jsonify({"error": "No files provided"}), 400

    files = request.files.getlist("files")
    results = []

    for file in files:
        if not file or not file.filename:
            continue
        if not _allowed_file(file.filename):
            results.append({"filename": file.filename, "error": "Not a PDF file"})
            continue

        filename = secure_filename(file.filename)
        save_path = os.path.join(Config.UPLOAD_FOLDER, filename)
        file.save(save_path)

        try:
            engine = _get_engine()
            info = engine.ingest_pdf(save_path)
            results.append(info)
        except Exception as e:
            results.append({"filename": filename, "error": str(e)})

    return jsonify({"results": results})


# ── Chat ─────────────────────────────────────────────────────────────────

@api_bp.route("/api/chat", methods=["POST"])
def chat():
    """
    POST /api/chat
    Body: {"message": "...", "session_id": "..." (optional)}
    """
    data = request.get_json(silent=True) or {}
    message = data.get("message", "").strip()

    if not message:
        return jsonify({"error": "Message is required"}), 400

    session_id = data.get("session_id", str(uuid.uuid4()))
    engine = _get_engine()
    result = engine.ask(message, session_id=session_id)
    result["session_id"] = session_id

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
    engine = _get_engine()

    def generate():
        for chunk in engine.ask_stream(message, session_id=session_id):
            yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"

    return Response(generate(), mimetype="text/event-stream")


# ── History ──────────────────────────────────────────────────────────────

@api_bp.route("/api/history/clear", methods=["POST"])
def clear_session_history():
    data = request.get_json(silent=True) or {}
    session_id = data.get("session_id", "")
    if session_id:
        clear_history(session_id)
    return jsonify({"status": "ok"})


# ── Index management ────────────────────────────────────────────────────

@api_bp.route("/api/index/status", methods=["GET"])
def index_status():
    engine = _get_engine()
    return jsonify({"chunks_indexed": engine.document_count})


@api_bp.route("/api/index/clear", methods=["POST"])
def clear_index():
    engine = _get_engine()
    engine.clear_index()
    return jsonify({"status": "ok"})
