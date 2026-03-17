from src.summarizer import DocumentSummarizer


class _FakeSummarizationPipeline:
    def __call__(self, text, max_length, min_length, do_sample, truncation):
        words = text.split()
        snippet = " ".join(words[: min(12, len(words))])
        return [{"summary_text": snippet or "empty summary"}]


def _fake_pipeline(*args, **kwargs):
    return _FakeSummarizationPipeline()


def test_hierarchical_summarize_returns_stable_contract(monkeypatch):
    monkeypatch.setattr(DocumentSummarizer, "_build_pipeline", _fake_pipeline)

    summarizer = DocumentSummarizer(model_name="dummy-model", max_length=40, min_length=10)
    chunks = [
        "This is the first chunk with enough words to summarize effectively.",
        "This is the second chunk with enough words to test aggregation behavior.",
    ]

    result = summarizer.hierarchical_summarize(chunks)

    assert isinstance(result, dict)
    assert "final_summary" in result
    assert "chunk_summaries" in result
    assert "num_chunks" in result
    assert len(result["chunk_summaries"]) == result["num_chunks"]


def test_summarize_long_document_includes_chunk_summaries(monkeypatch):
    monkeypatch.setattr(DocumentSummarizer, "_build_pipeline", _fake_pipeline)

    summarizer = DocumentSummarizer(model_name="dummy-model", max_length=40, min_length=10)
    text = (
        "Long document text with multiple segments to verify output schema across methods. "
        "The text should generate non-empty summaries and include metadata fields."
    )
    chunks = [
        "Chunk one has meaningful content for summarization to happen.",
        "Chunk two has additional context for generating a final summary.",
    ]

    hierarchical = summarizer.summarize_long_document(
        text=text, chunks=chunks, method="hierarchical"
    )
    concatenate = summarizer.summarize_long_document(text=text, chunks=chunks, method="concatenate")

    for result in (hierarchical, concatenate):
        assert "final_summary" in result
        assert "chunk_summaries" in result
        assert "compression_ratio" in result
        assert result["summary_length"] >= 1
        assert isinstance(result["chunk_summaries"], list)


def test_rag_summarize_includes_retrieval_metadata(monkeypatch):
    monkeypatch.setattr(DocumentSummarizer, "_build_pipeline", _fake_pipeline)

    summarizer = DocumentSummarizer(model_name="dummy-model", max_length=40, min_length=10)
    text = " ".join(
        [
            "This paper introduces a transformer architecture for long document summarization.",
            "It presents retrieval-augmented strategies to keep relevant context.",
            "An unrelated sentence about cooking appears in this document.",
        ]
    )
    chunks = [
        "This paper introduces a transformer architecture for long document summarization.",
        "It presents retrieval-augmented strategies to keep relevant context.",
        "An unrelated sentence about cooking appears in this document.",
    ]

    result = summarizer.summarize_long_document(
        text=text,
        chunks=chunks,
        method="rag",
        rag_query="retrieval augmented summarization transformer",
        retrieval_top_k=2,
    )

    assert result["method"] == "rag"
    assert "retrieval_query" in result
    assert "retrieved_chunk_ids" in result
    assert "retrieved_scores" in result
    assert isinstance(result["retrieved_chunk_ids"], list)
    assert isinstance(result["retrieved_scores"], list)
