from src.retriever import ChunkRetriever


def test_retrieve_returns_relevant_chunk_first():
    chunks = [
        "This section discusses climate change and carbon emissions.",
        "Neural summarization models use transformers and attention mechanisms.",
        "Cooking recipes involve ingredients, heat, and timing.",
    ]

    retriever = ChunkRetriever(chunks)
    retrieved = retriever.retrieve("transformers summarization", top_k=2)

    assert len(retrieved) >= 1
    assert retrieved[0].chunk_id == 1
    assert retrieved[0].score > 0


def test_retrieve_handles_empty_query():
    retriever = ChunkRetriever(["sample chunk"])
    retrieved = retriever.retrieve("", top_k=3)

    assert retrieved == []
