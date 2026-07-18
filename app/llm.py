"""LLM integration via Groq for generating answers."""

import os
from groq import Groq


def build_prompt(query: str, context_chunks: list[dict], history: list[dict] | None = None) -> list[dict]:
    """
    Build the message list for the LLM.
    """
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
            "4. Do not refuse to answer unless the documents are completely unrelated. Be completely case-insensitive (e.g., treat 'api' the same as 'API', or 'apple' the same as 'Apple').\n"
            "5. CRITICAL: Always respond in the EXACT same language that the user asks their question in.\n\n"
            f"--- CONTEXT ---\n{context_block}\n--- END CONTEXT ---"
        ),
    }

    messages = [system_message]

    if history:
        messages.extend(history)

    messages.append({"role": "user", "content": query})

    return messages


def query_llm(
    query: str,
    context_chunks: list[dict],
    history: list[dict] | None = None,
    model: str = "llama-3.1-8b-instant",
    api_key: str | None = None,
    **kwargs
) -> str:
    """Send query to Groq."""
    if not api_key:
        api_key = os.getenv("GROQ_API_KEY")
        
    client = Groq(api_key=api_key)
    messages = build_prompt(query, context_chunks, history)

    response = client.chat.completions.create(
        model=model, 
        messages=messages,
        temperature=0.2
    )
    return response.choices[0].message.content


def query_llm_stream(
    query: str,
    context_chunks: list[dict],
    history: list[dict] | None = None,
    model: str = "llama-3.1-8b-instant",
    api_key: str | None = None,
    **kwargs
):
    """Stream query to Groq."""
    if not api_key:
        api_key = os.getenv("GROQ_API_KEY")
        
    client = Groq(api_key=api_key)
    messages = build_prompt(query, context_chunks, history)

    stream = client.chat.completions.create(
        model=model, 
        messages=messages, 
        stream=True,
        temperature=0.2
    )
    
    for chunk in stream:
        token = chunk.choices[0].delta.content
        if token:
            yield token
