"""LLM integration via Universal OpenAI Client for generating answers."""

import os
import re
from openai import OpenAI

# Connection pool cache: (api_key, base_url) -> OpenAI instance
_clients: dict[tuple[str, str], OpenAI] = {}
_THINK_RE = re.compile(r'<think>.*?</think>', flags=re.DOTALL)


def _get_client(api_key: str | None, base_url: str | None) -> tuple[OpenAI, str]:
    """Retrieve or create a cached OpenAI client for HTTP connection reuse."""
    if not api_key:
        api_key = os.getenv("LLM_API_KEY", "")
    if not base_url:
        base_url = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")

    key = (api_key, base_url)
    if key not in _clients:
        _clients[key] = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=60.0,
            max_retries=2,
        )
    return _clients[key]


def build_prompt(
    query: str,
    context_chunks: list[dict],
    history: list[dict] | None = None,
    system_prompt: str | None = None,
) -> list[dict]:
    """Build the message list for the LLM."""
    context_parts = []
    for i, chunk in enumerate(context_chunks, 1):
        source = chunk.get("source", "unknown")
        page = chunk.get("page", "?")
        context_parts.append(f"[{i}] (Source: {source}, Page {page})\n{chunk['text']}")

    context_block = "\n\n".join(context_parts)

    default_system = (
        "You are an expert, highly intelligent AI assistant and researcher. "
        "Your task is to answer the user's question accurately, thoroughly, and clearly, using ONLY the provided document context.\n\n"
        "Rules you must follow:\n"
        "1. Read the provided context carefully before answering.\n"
        "2. Think step-by-step and structure your answer logically with bullet points or paragraphs if it's long.\n"
        "3. If the exact answer is not in the context, use the documents to provide the best possible related information.\n"
        "4. Do not refuse to answer unless the documents are completely unrelated. Be completely case-insensitive.\n"
        "5. CRITICAL: Always respond in the EXACT same language that the user asks their question in.\n"
        "6. If you cannot find the answer in the provided documents, state clearly that the answer is not in the knowledge base.\n\n"
        f"--- CONTEXT ---\n{context_block}\n--- END CONTEXT ---"
    )

    if system_prompt:
        system_content = f"{system_prompt}\n\n--- CONTEXT ---\n{context_block}\n--- END CONTEXT ---"
    else:
        system_content = default_system

    messages = [{"role": "system", "content": system_content}]

    if history:
        messages.extend(history)

    messages.append({"role": "user", "content": query})
    return messages


def query_llm(
    query: str,
    context_chunks: list[dict],
    history: list[dict] | None = None,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    system_prompt: str | None = None,
    temperature: float = 0.2,
    **kwargs
) -> str:
    """Send query to the LLM API using connection pooling."""
    if not model:
        model = os.getenv("LLM_MODEL", "qwen/qwen3.6-27b")

    client = _get_client(api_key, base_url)
    messages = build_prompt(query, context_chunks, history, system_prompt=system_prompt)

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    content = response.choices[0].message.content or ""
    return _THINK_RE.sub('', content).strip()


def query_llm_stream(
    query: str,
    context_chunks: list[dict],
    history: list[dict] | None = None,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    system_prompt: str | None = None,
    temperature: float = 0.2,
    **kwargs
):
    """Stream query to the LLM API using connection pooling."""
    if not model:
        model = os.getenv("LLM_MODEL", "qwen/qwen3.6-27b")

    client = _get_client(api_key, base_url)
    messages = build_prompt(query, context_chunks, history, system_prompt=system_prompt)

    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
        temperature=temperature,
    )

    in_think = False
    for chunk in stream:
        if len(chunk.choices) > 0 and chunk.choices[0].delta.content:
            token = chunk.choices[0].delta.content
            if "<think>" in token:
                in_think = True
                continue
            if "</think>" in token:
                in_think = False
                continue
            if not in_think:
                yield token

