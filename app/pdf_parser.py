"""Document text extraction for PDFs and plain text files."""

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


def extract_text_from_txt(txt_path: str) -> list[dict]:
    """
    Extract text from a plain text (.txt) file.

    Returns:
        [{"page": 1, "text": "...", "source": "filename.txt"}]
    """
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"File not found: {txt_path}")

    filename = os.path.basename(txt_path)
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
    except UnicodeDecodeError:
        with open(txt_path, "r", encoding="latin-1", errors="ignore") as f:
            text = f.read().strip()

    if not text:
        return []

    return [{
        "page": 1,
        "text": text,
        "source": filename,
    }]


def extract_text_from_file(file_path: str) -> list[dict]:
    """Extract text based on file extension (.pdf or .txt)."""
    ext = file_path.rsplit(".", 1)[-1].lower() if "." in file_path else ""
    if ext == "pdf":
        return extract_text_from_pdf(file_path)
    elif ext == "txt":
        return extract_text_from_txt(file_path)
    else:
        raise ValueError(f"Unsupported file format: .{ext}")


def extract_text_from_multiple(file_paths: list[str]) -> list[dict]:
    """Extract text from multiple files."""
    all_pages = []
    for path in file_paths:
        all_pages.extend(extract_text_from_file(path))
    return all_pages
