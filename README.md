# RAG Chat — AI Document Assistant

A free, fully local Retrieval-Augmented Generation (RAG) chat application that allows you to talk to your PDF documents. It uses **Flask** for the backend, **FAISS** for vector search, and **Ollama** to run large language models completely offline.

![RAG Chat UI](https://via.placeholder.com/800x450.png?text=RAG+Chat+UI+Screenshot)

## ✨ Features
* **100% Free & Local:** No cloud API keys needed. Your documents stay entirely on your machine.
* **Smart PDF Parsing:** Extracts text from PDFs using PyMuPDF and intelligently chunks it by sentences.
* **Vector Search:** Uses `sentence-transformers` and FAISS to instantly find relevant context.
* **Beautiful UI:** A modern, glassmorphic dark-mode web interface.
* **Markdown Support:** The AI's responses are streamed in real-time and formatted beautifully with lists, bold text, and code blocks.
* **Multi-lingual:** Automatically answers in the exact same language you ask your questions in.
* **Source Citations:** Shows exactly which pages it pulled information from.

## 🛠️ Tech Stack
* **Backend:** Python, Flask
* **AI & LLMs:** Ollama (running `qwen2.5:3b` by default)
* **Embeddings:** `sentence-transformers` (`all-MiniLM-L6-v2`)
* **Vector Database:** FAISS (CPU)
* **Frontend:** HTML5, Vanilla CSS, JS (with `marked.js` for markdown)

---

## 🚀 Getting Started

### Prerequisites
1. **Python 3.10+** installed on your machine.
2. **[Ollama](https://ollama.com/)** installed and running on your machine.

### 1. Download the AI Model
Open your terminal and pull the default lightweight model (Qwen 2.5 3B):
```bash
ollama pull qwen2.5:3b
```
*(You can also use `llama3.2`, `mistral`, or any other model supported by Ollama!)*

### 2. Setup the Project
Clone the repository and set up your virtual environment:

```bash
# Clone the repository
git clone https://github.com/yourusername/rag-chat.git
cd rag-chat

# Create a virtual environment
python -m venv venv

# Activate it (Windows)
venv\Scripts\activate
# Activate it (Mac/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Variables
Copy the example environment file:
```bash
cp .env.example .env
```
*(You can edit the `.env` file to change the default chunk size, Ollama model, or Flask port).*

### 4. Run the App
```bash
python run.py
```
Open your browser and navigate to `http://localhost:5000`. Drag and drop a PDF into the sidebar, and start chatting!

---

## 🛑 Common Issues & Troubleshooting

**1. "I get an SSL Certificate Error when trying to upload a PDF"**
If you see an error mentioning `CERTIFICATE_VERIFY_FAILED` on Windows, your Python cannot verify the HuggingFace servers to download the embedding model.
* **Fix:** Run `pip install --upgrade certifi` or `python -m pip install python-certifi-win32` in your terminal.

**2. "The AI says it doesn't have any documents to search"**
Ensure that your PDF actually contains extractable text (and is not just scanned images). If the document is purely images, the current PDF parser (PyMuPDF) will not extract OCR data by default.

**3. "The AI takes a very long time to answer"**
The app is configured to keep the model loaded in memory permanently (`keep_alive=-1`). However, if your computer has limited RAM, Ollama might struggle. Ensure you have at least 8GB of RAM for the 3B models, and close other heavy applications.

---

## 📂 Project Structure
```text
rag-chat/
├── app/
│   ├── config.py          # Environment settings
│   ├── routes.py          # Flask REST APIs
│   ├── rag_engine.py      # Core RAG logic & pipeline
│   ├── vector_store.py    # FAISS database manager
│   ├── llm.py             # Ollama integration
│   ├── embedder.py        # Sentence transformers
│   ├── chunker.py         # Text chunking logic
│   └── pdf_parser.py      # PDF text extraction
├── static/
│   ├── css/style.css      # Dark-mode UI styling
│   └── js/app.js          # Chat, streaming, and UI logic
├── templates/
│   └── index.html         # Main web interface
├── data/                  # Auto-generated FAISS database
├── uploads/               # Temporary PDF storage
├── requirements.txt
└── run.py                 # Application entry point
```

## 📜 License
This project is open-source and available under the MIT License. Feel free to fork it, modify it, and use it for your own personal local AI setup!
