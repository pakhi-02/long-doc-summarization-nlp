from src.chunker import TextChunker, split_long_document


def test_chunk_text_sentences_produces_multiple_chunks():
    text = (
        "Sentence one. Sentence two is a bit longer. Sentence three continues the story. "
        "Sentence four adds more content. Sentence five wraps it up."
    )
    chunker = TextChunker(chunk_size=8, overlap=2)

    chunks = chunker.chunk_text(text, method="sentences")

    assert len(chunks) >= 2
    assert all(isinstance(chunk, str) and chunk.strip() for chunk in chunks)


def test_split_long_document_returns_metadata():
    text = " ".join([f"word{i}" for i in range(120)])

    chunks, metadata = split_long_document(text, max_chunk_size=30, overlap=5, method="tokens")

    assert len(chunks) == len(metadata)
    assert len(chunks) > 1
    assert all("word_count" in m and "chunk_id" in m for m in metadata)
