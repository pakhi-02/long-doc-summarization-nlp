# Quick Start Guide

## 🎯 Goal
Summarize long documents that exceed transformer model context limits using chunking and hierarchical summarization.

## ⚡ 3-Minute Setup

```bash
# 1. Setup environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Test the installation
pytest -q

# 3. Run the web app
streamlit run app.py
```

## 📝 Basic Usage

### Web Interface
1. Run `streamlit run app.py`
2. Upload a PDF or paste text
3. Adjust parameters in sidebar
4. Click "Generate Summary"
5. View results and metrics

### Python Code
```python
from src.chunker import split_long_document
from src.summarizer import DocumentSummarizer

# Your long text
text = "..." # Your document here

# Chunk it
chunks, _ = split_long_document(text)

# Summarize
summarizer = DocumentSummarizer()
result = summarizer.hierarchical_summarize(chunks)

print(result['final_summary'])
```

## 🔧 Key Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| chunk_size | Words per chunk | 1024 | 256-2048 |
| overlap | Overlap between chunks | 128 | 0-512 |
| max_length | Max summary length | 150 | 50-500 |
| method | Chunking strategy | sentences | tokens/sentences/paragraphs |

## 🎓 When to Use Each Method

- **Hierarchical** (Recommended): Best for very long documents (>5000 words)
- **Concatenate**: Faster but less coherent for very long docs
- **RAG**: Best when you care about specific aspects (e.g., methods, results, limitations)
- **Sentences**: Best for most cases, preserves meaning
- **Paragraphs**: Good for well-structured documents
- **Tokens**: Simple word-based splitting

## 🔎 RAG Example

```python
from src.chunker import split_long_document
from src.summarizer import DocumentSummarizer

chunks, _ = split_long_document(text)
summarizer = DocumentSummarizer()

result = summarizer.summarize_long_document(
	text=text,
	chunks=chunks,
	method="rag",
	rag_query="main contributions and experimental results",
	retrieval_top_k=5,
)

print(result["final_summary"])
print(result["retrieved_chunk_ids"])
```

## 📊 Understanding Metrics

- **ROUGE-1**: Unigram overlap (0-1, higher better)
- **ROUGE-L**: Longest common subsequence
- **BERTScore**: Semantic similarity (0-1, higher better)
- **Compression Ratio**: Original length / Summary length

## ⚠️ Common Issues

1. **Out of Memory**: Reduce chunk_size or use smaller model (t5-small)
2. **Slow Processing**: Use GPU or reduce max_length
3. **Poor Quality**: Increase chunk_size or try different model

## 🧪 Optional Full Pipeline Check

```bash
python test_pipeline.py
```

## 🚀 Next Steps

- Try different models: BART, T5, PEGASUS
- Experiment with chunk sizes
- Use reference summaries for evaluation
- Fine-tune on domain-specific data

## 💡 Tips

- Start with default parameters
- Use sentence-based chunking for best results
- Larger overlap preserves context but slower
- BART generally works best for news/general text
