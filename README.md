# RAG Chat — AI Document Assistant

A free, fully local Retrieval-Augmented Generation (RAG) chat application that allows you to talk to your PDF documents. It uses **Flask** for the backend, **FAISS** for vector search, and **Ollama** to run large language models completely offline.



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
* **AI & LLMs:** Groq API (`llama-3.1-8b-instant`)
* **Embeddings:** `sentence-transformers` (`all-MiniLM-L6-v2`)
* **Vector Database:** FAISS (CPU)
* **Frontend:** HTML5, Vanilla CSS, JS (with `marked.js` for markdown)
* **Hosting:** Docker (ready for Hugging Face Spaces)

---

## 🚀 Getting Started Locally

### 1. Setup the Project
Clone the repository and set up your virtual environment:

```bash
git clone https://github.com/yourusername/rag-chat.git
cd rag-chat
python -m venv venv
venv\Scripts\activate  # On Windows
pip install -r requirements.txt
```

### 2. Get a Groq API Key
1. Go to the [Groq Console](https://console.groq.com/keys) and create a free account.
2. Generate an API Key.
3. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
4. Paste your API key inside `.env`:
   ```
   GROQ_API_KEY=gsk_your_api_key_here
   ```

### 3. Run the App
```bash
python run.py
```
Open your browser and navigate to `http://localhost:5000`.

---

## ☁️ Deploying Online (Free 24/7 Hosting)

You can host this app completely for free using **Hugging Face Spaces**. It will use 0% of your computer's memory!

1. Create a free account on [Hugging Face](https://huggingface.co/).
2. Go to your profile and click **New Space**.
3. Set a name for your space (e.g., `my-rag-chat`).
4. Select **Docker** as the Space SDK, and choose **Blank**.
5. Set Space Hardware to **Free**.
6. Once the space is created, click **Settings** -> **Variables and secrets**.
   - Add a new **Secret** named `GROQ_API_KEY` and paste your Groq API key.
7. Go back to the **App** tab, click **Files**, and upload all the files from this folder (including the `Dockerfile`).
8. Hugging Face will automatically build your Docker container and start your app!

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
