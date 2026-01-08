from PyPDF2 import PdfReader
import re


def load_pdf(file_path: str) -> str:
    """
    Load a PDF file and extract text from all pages.

    Args:
        file_path (str): Path to the PDF file

    Returns:
        str: Extracted raw text
    """
    reader = PdfReader(file_path)
    text = []

    for page_num, page in enumerate(reader.pages):
        try:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
        except Exception as e:
            print(f"Warning: Could not extract text from page {page_num}: {e}")

    return "\n".join(text)


def clean_text(text: str) -> str:
    """
    Clean extracted text by removing excessive whitespace
    and normalizing formatting.

    Args:
        text (str): Raw extracted text

    Returns:
        str: Cleaned text
    """
    # Remove multiple newlines
    text = re.sub(r"\n{2,}", "\n", text)

    # Remove extra spaces
    text = re.sub(r"[ \t]+", " ", text)

    # Remove trailing spaces
    text = text.strip()

    return text


def load_and_clean_pdf(file_path: str) -> str:
    """
    Load a PDF and return cleaned text.

    Args:
        file_path (str): Path to the PDF file

    Returns:
        str: Cleaned document text
    """
    raw_text = load_pdf(file_path)
    cleaned_text = clean_text(raw_text)
    return cleaned_text
