"""LLM integration via Ollama for generating answers."""

import ollama


def build_prompt(query: str, context_chunks: list[dict], history: list[dict] | None = None) -> list[dict]:
    """
    Build the message list for the LLM.

    Args:
        query: The user's question.
        context_chunks: Retrieved chunks with 'text', 'source', 'page' keys.
        history: Previous conversation turns [{"role": "user"/"assistant", "content": "..."}].

    Returns:
        List of message dicts for the Ollama chat API.
    """
    # Format context block
    context_parts = []
    for i, chunk in enumerate(context_chunks, 1):
        source = chunk.get("source", "unknown")
        page = chunk.get("page", "?")
        context_parts.append(f"[{i}] (Source: {source}, Page {page})\n{chunk['text']}")

    context_block = "\n\n".join(context_parts)

    system_message = {
        "role": "system",
        "content": (
            "You are an expert, highly intelligent AI assistant and researcher. "
            "Your task is to answer the user's question accurately, thoroughly, and clearly, using ONLY the provided document context.\n\n"
            "Rules you must follow:\n"
            "1. Read the provided context carefully before answering.\n"
            "2. Think step-by-step and structure your answer logically with bullet points or paragraphs if it's long.\n"
            "3. If the exact answer is not in the context, use the documents to provide the best possible related information.\n"
            "4. Do not refuse to answer unless the documents are completely unrelated.\n"
            "5. CRITICAL: Always respond in the EXACT same language that the user asks their question in (e.g., if the user asks in Arabic, you MUST reply in Arabic).\n\n"
            f"--- CONTEXT ---\n{context_block}\n--- END CONTEXT ---"
        ),
    }

    messages = [system_message]

    # Add conversation history
    if history:
        messages.extend(history)

    messages.append({"role": "user", "content": query})

    return messages


def query_llm(
    query: str,
    context_chunks: list[dict],
    history: list[dict] | None = None,
    model: str = "llama3",
    base_url: str = "http://localhost:11434",
) -> str:
    """
    Send query + context to the Ollama LLM and return the response.

    Args:
        query: User's question.
        context_chunks: Retrieved document chunks.
        history: Conversation history.
        model: Ollama model name.
        base_url: Ollama server URL.

    Returns:
        The LLM's response text.
    """
    client = ollama.Client(host=base_url)
    messages = build_prompt(query, context_chunks, history)

    response = client.chat(
        model=model, 
        messages=messages,
        keep_alive=-1, # Keeps the model loaded in memory permanently
        options={"temperature": 0.2}  # Lower temp makes answers more factual and logical
    )
    return response["message"]["content"]


def query_llm_stream(
    query: str,
    context_chunks: list[dict],
    history: list[dict] | None = None,
    model: str = "llama3",
    base_url: str = "http://localhost:11434",
):
    """
    Stream query + context to the Ollama LLM, yielding response chunks.

    Yields:
        str: Chunks of the LLM's response text.
    """
    client = ollama.Client(host=base_url)
    messages = build_prompt(query, context_chunks, history)

    stream = client.chat(
        model=model, 
        messages=messages, 
        stream=True,
        keep_alive=-1, # Keeps the model loaded in memory permanently
        options={"temperature": 0.2}
    )
    for chunk in stream:
        token = chunk["message"]["content"]
        if token:
            yield token
