# RAG Chat — Headless REST API

A high-performance, fully headless Retrieval-Augmented Generation (RAG) REST API service. Chat with your documents, ingest files or raw text, perform ultra-fast hybrid search, and manage vector indices completely through standard HTTP APIs.

## ⚡ Features & Performance
* **100% Headless:** Pure REST API architecture. No UI bloat, easily integrable into n8n, workflows, web apps, or backend services.
* **Ultra-Fast Hybrid Search:** Combines FAISS L2 vector embeddings with optimized exact pattern/keyword matching (sub-60ms retrieval).
* **Connection Pooling:** Cached LLM client sessions with HTTP keep-alive reuse for minimum time-to-first-token.
* **Flexible Ingestion:** Ingest `.pdf`, `.txt`, binary files, or raw text directly via JSON payloads.
* **Real-Time Streaming:** Server-Sent Events (SSE) streaming endpoint for instantaneous token streaming.
* **Document & History Management:** Full API control over indexed documents, chunk statistics, and multi-session conversation history.

---

## 🛠️ Tech Stack
* **Framework:** Python, Flask (Headless)
* **LLM Engine:** Universal OpenAI Standard (Groq, DashScope/Qwen, OpenRouter, Ollama, etc.)
* **Embeddings:** `sentence-transformers` (`all-MiniLM-L6-v2`) with PyTorch inference optimization
* **Vector Store:** FAISS (CPU)
* **PDF Extraction:** PyMuPDF (`fitz`)

---

## 🚀 Quick Start

### 1. Installation
```bash
git clone https://github.com/yourusername/rag-chat.git
cd rag-chat
python -m venv venv
venv\Scripts\activate  # On Windows (or source venv/bin/activate on Linux/Mac)
pip install -r requirements.txt
```

### 2. Configure Environment (`.env`)
Copy `.env.example` to `.env` and set your LLM API credentials:
```env
LLM_API_KEY=your_api_key_here
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_MODEL=qwen/qwen3.6-27b
```

### 3. Run the API Server
```bash
python run.py
```
Server will be available at `http://localhost:5000`.

---

## 📡 REST API Reference

### 1. Health & Endpoint Catalog
`GET /` or `GET /api` or `GET /health`

**Response:**
```json
{
  "status": "online",
  "service": "RAG Chat Headless REST API",
  "version": "2.0.0",
  "config": {
    "embedding_model": "all-MiniLM-L6-v2",
    "llm_model": "qwen/qwen3.6-27b"
  },
  "stats": {
    "chunks_indexed": 12,
    "documents_indexed": 2
  }
}
```

---

### 2. Ingest Document File
`POST /api/upload`

Upload `.pdf` or `.txt` files using multipart form-data or raw binary.

**cURL Example:**
```bash
curl -X POST http://localhost:5000/api/upload \
  -F "file=@/path/to/document.pdf"
```

---

### 3. Ingest Raw Text / Snippet
`POST /api/documents/text`

Directly embed and index a raw text snippet or markdown notes without creating files.

**Request Body:**
```json
{
  "text": "Antigravity Superchargers are located in Sector 7 and provide 350kW ultra-fast charging.",
  "title": "chargers_guide.txt"
}
```

**Response:**
```json
{
  "result": {
    "filename": "chargers_guide.txt",
    "pages": 1,
    "chunks": 1
  },
  "total_chunks_indexed": 1
}
```

---

### 4. Fast Hybrid Search (Without LLM)
`POST /api/search`

Perform rapid similarity and keyword retrieval against indexed chunks.

**Request Body:**
```json
{
  "query": "Where is the 350kW supercharger?",
  "top_k": 3
}
```

**Response:**
```json
{
  "query": "Where is the 350kW supercharger?",
  "count": 1,
  "latency_ms": 58.14,
  "results": [
    {
      "source": "chargers_guide.txt",
      "page": 1,
      "chunk_id": 0,
      "text": "Antigravity Superchargers are located in Sector 7 and provide 350kW ultra-fast charging.",
      "score": 0.0025
    }
  ]
}
```

---

### 5. Chat with RAG (JSON Response)
`POST /api/chat`

**Request Body:**
```json
{
  "message": "Where can I find the 350kW supercharger?",
  "session_id": "user_session_123",
  "top_k": 5,
  "temperature": 0.2
}
```

**Response:**
```json
{
  "answer": "The 350kW superchargers are located in Sector 7.",
  "session_id": "user_session_123",
  "latency_ms": 320.5,
  "sources": [
    {
      "source": "chargers_guide.txt",
      "page": 1
    }
  ]
}
```

---

### 6. Chat Streaming (Server-Sent Events)
`POST /api/chat/stream`

Streams tokens in real-time.

**cURL Example:**
```bash
curl -N -X POST http://localhost:5000/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "Summarize the key points", "session_id": "sess_1"}'
```

---

### 7. List Indexed Documents
`GET /api/documents`

**Response:**
```json
{
  "documents": [
    {
      "source": "chargers_guide.txt",
      "chunks": 1,
      "page_count": 1
    }
  ],
  "total_documents": 1,
  "total_chunks": 1
}
```

---

### 8. Delete a Document
`DELETE /api/documents/<filename>`

Removes all vectors associated with the specified document from the index.

**cURL Example:**
```bash
curl -X DELETE http://localhost:5000/api/documents/chargers_guide.txt
```

---

### 9. Get / Clear Conversation History
* `GET /api/history?session_id=user_session_123` — Retrieve history for a session.
* `POST /api/history/clear` with `{"session_id": "user_session_123"}` — Clear history.

---

### 10. Vector Index Status & Clear
* `GET /api/index/status` — Get chunk & document counts.
* `POST /api/index/clear` — Clear entire vector index.

---

## 📂 Headless Architecture Structure
```text
rag-chat/
├── app/
│   ├── config.py          # Environment configuration
│   ├── routes.py          # REST API endpoints & request handlers
│   ├── rag_engine.py      # Core RAG orchestration pipeline
│   ├── vector_store.py    # FAISS vector store & hybrid search engine
│   ├── llm.py             # OpenAI client pooling & generation
│   ├── embedder.py        # Sentence transformers with PyTorch inference mode
│   ├── chunker.py         # Smart sentence-boundary text chunker
│   ├── pdf_parser.py      # PDF & text extractors
│   └── chat_history.py    # Multi-session chat history store
├── data/                  # FAISS index storage
├── uploads/               # Temporary file storage
├── Dockerfile             # Container definition for cloud/HuggingFace deployment
├── requirements.txt       # Python dependencies
├── run.py                 # Application entry point
└── test_api.py            # API automated test suite
```
