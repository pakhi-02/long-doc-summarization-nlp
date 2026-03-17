"""
Text chunking module for splitting long documents into manageable pieces.
Supports overlapping chunks to preserve context across boundaries.
"""

from typing import List, Optional, Tuple
import re


class TextChunker:
    """
    Splits long documents into smaller chunks with optional overlap.
    """

    def __init__(self, chunk_size: int = 1024, overlap: int = 128, preserve_sentences: bool = True):
        """
        Initialize the TextChunker.

        Args:
            chunk_size (int): Target size of each chunk in tokens/characters
            overlap (int): Number of tokens/characters to overlap between chunks
            preserve_sentences (bool): Try to split at sentence boundaries
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.preserve_sentences = preserve_sentences

    def chunk_by_tokens(self, text: str, max_tokens: Optional[int] = None) -> List[str]:
        """
        Split text into chunks based on token count (approximate using words).

        Args:
            text (str): Input text to chunk
            max_tokens (int): Maximum tokens per chunk (overrides chunk_size)

        Returns:
            List[str]: List of text chunks
        """
        max_tokens = max_tokens or self.chunk_size
        words = text.split()
        chunks: List[str] = []

        i = 0
        while i < len(words):
            # Get chunk of words
            chunk_words = words[i : i + max_tokens]
            chunk = " ".join(chunk_words)
            chunks.append(chunk)

            # Move forward with overlap
            i += max_tokens - self.overlap

            if i >= len(words):
                break

        return chunks

    def chunk_by_sentences(self, text: str) -> List[str]:
        """
        Split text into chunks while preserving sentence boundaries.

        Args:
            text (str): Input text to chunk

        Returns:
            List[str]: List of text chunks
        """
        # Split into sentences (simple regex-based)
        sentences = re.split(r"(?<=[.!?])\s+", text)

        chunks: List[str] = []
        current_chunk: List[str] = []
        current_size = 0

        for sentence in sentences:
            sentence_size = len(sentence.split())

            # If adding this sentence exceeds chunk size, save current chunk
            if current_size + sentence_size > self.chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))

                # Start new chunk with overlap from previous
                if self.overlap > 0:
                    overlap_sentences: List[str] = []
                    overlap_size = 0
                    for s in reversed(current_chunk):
                        overlap_size += len(s.split())
                        if overlap_size >= self.overlap:
                            break
                        overlap_sentences.insert(0, s)
                    current_chunk = overlap_sentences
                    current_size = overlap_size
                else:
                    current_chunk = []
                    current_size = 0

            current_chunk.append(sentence)
            current_size += sentence_size

        # Add final chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def chunk_by_paragraphs(self, text: str) -> List[str]:
        """
        Split text into chunks at paragraph boundaries.

        Args:
            text (str): Input text to chunk

        Returns:
            List[str]: List of text chunks
        """
        paragraphs = text.split("\n\n")
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        chunks: List[str] = []
        current_chunk: List[str] = []
        current_size = 0

        for para in paragraphs:
            para_size = len(para.split())

            if current_size + para_size > self.chunk_size and current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_size = 0

            current_chunk.append(para)
            current_size += para_size

        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        return chunks

    def chunk_text(self, text: str, method: str = "sentences") -> List[str]:
        """
        Main method to chunk text using specified strategy.

        Args:
            text (str): Input text to chunk
            method (str): Chunking method - "tokens", "sentences", or "paragraphs"

        Returns:
            List[str]: List of text chunks
        """
        if method == "tokens":
            return self.chunk_by_tokens(text)
        elif method == "sentences":
            return self.chunk_by_sentences(text)
        elif method == "paragraphs":
            return self.chunk_by_paragraphs(text)
        else:
            raise ValueError(f"Unknown chunking method: {method}")

    def get_chunk_metadata(self, chunks: List[str]) -> List[dict]:
        """
        Get metadata about each chunk.

        Args:
            chunks (List[str]): List of text chunks

        Returns:
            List[dict]: Metadata for each chunk (index, size, etc.)
        """
        metadata = []
        for i, chunk in enumerate(chunks):
            metadata.append(
                {
                    "chunk_id": i,
                    "word_count": len(chunk.split()),
                    "char_count": len(chunk),
                    "preview": chunk[:100] + "..." if len(chunk) > 100 else chunk,
                }
            )
        return metadata


def split_long_document(
    text: str, max_chunk_size: int = 1024, overlap: int = 128, method: str = "sentences"
) -> Tuple[List[str], List[dict]]:
    """
    Convenience function to split a long document into chunks.

    Args:
        text (str): Input document text
        max_chunk_size (int): Maximum size per chunk
        overlap (int): Overlap between chunks
        method (str): Chunking strategy

    Returns:
        Tuple[List[str], List[dict]]: Chunks and their metadata
    """
    chunker = TextChunker(chunk_size=max_chunk_size, overlap=overlap)
    chunks = chunker.chunk_text(text, method=method)
    metadata = chunker.get_chunk_metadata(chunks)
    return chunks, metadata
