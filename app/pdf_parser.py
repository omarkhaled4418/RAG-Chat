"""PDF text extraction using PyMuPDF."""

import os
import fitz  # PyMuPDF


def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """
    Extract text from a PDF file, returning a list of dicts with page content
    and metadata.

    Returns:
        [{"page": 1, "text": "...", "source": "filename.pdf"}, ...]
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    filename = os.path.basename(pdf_path)
    pages = []

    doc = fitz.open(pdf_path)
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()
        if text:
            pages.append({
                "page": page_num,
                "text": text,
                "source": filename,
            })
    doc.close()

    return pages


def extract_text_from_multiple(pdf_paths: list[str]) -> list[dict]:
    """Extract text from multiple PDF files."""
    all_pages = []
    for path in pdf_paths:
        all_pages.extend(extract_text_from_pdf(path))
    return all_pages
