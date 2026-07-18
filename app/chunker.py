"""Text chunking utilities for splitting extracted text into smaller pieces."""


def chunk_text(
    pages: list[dict],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[dict]:
    """
    Split page texts into overlapping chunks.

    Args:
        pages: List of dicts from pdf_parser (keys: page, text, source).
        chunk_size: Max number of characters per chunk.
        chunk_overlap: Number of overlapping characters between chunks.

    Returns:
        [{"text": "...", "source": "file.pdf", "page": 1, "chunk_id": 0}, ...]
    """
    chunks = []
    chunk_id = 0

    for page in pages:
        text = page["text"]
        source = page["source"]
        page_num = page["page"]

        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]

            # Try to break at a sentence boundary when possible
            if end < len(text):
                # Look for the last period, newline, or question mark
                for sep in [". ", ".\n", "? ", "?\n", "! ", "!\n", "\n\n"]:
                    last_sep = chunk.rfind(sep)
                    if last_sep > chunk_size // 2:
                        chunk = chunk[: last_sep + len(sep)]
                        end = start + len(chunk)
                        break

            chunk = chunk.strip()
            if chunk:
                chunks.append({
                    "text": chunk,
                    "source": source,
                    "page": page_num,
                    "chunk_id": chunk_id,
                })
                chunk_id += 1

            start = end - chunk_overlap

    return chunks
