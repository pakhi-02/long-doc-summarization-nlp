# Long Document Text Summarizer

This project implements a long-document text summarization system designed for domains such as legal, medical, and research documents. The system handles documents exceeding standard transformer context limits by applying chunking strategies and hierarchical summarization to generate accurate and faithful summaries.

It now also includes a lightweight retrieval-augmented generation (RAG) mode that first retrieves the most relevant chunks for a query, then summarizes only the retrieved context to reduce context dilution.

## Problem Statement
Transformer-based summarization models struggle with long documents due to context length limitations, often resulting in information loss or hallucinations. This project addresses these challenges using chunk-based and hierarchical summarization techniques.

## Quick Start

### Installation

1. **Clone the repository**
```bash
git clone <repo-url>
cd long-doc-summarization-nlp
```

2. **Create virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate (mac)  
On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Usage

#### Option 1: Streamlit Web App (Recommended)

Launch the interactive web application:

```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

**Features:**
- 📤 Upload PDF files or paste text
- ⚙️ Configure chunking and summarization parameters
- 📊 View evaluation metrics (ROUGE, BERTScore)
- 🔍 Check faithfulness and detect hallucinations

#### Option 2: Python Script

Use the summarization pipeline in your code:

```python
from src.chunker import split_long_document
from src.summarizer import DocumentSummarizer
from src.loader import load_pdf, clean_text

# Load your document
text = load_pdf("path/to/document.pdf")
text = clean_text(text)

# Chunk the document
chunks, metadata = split_long_document(
    text,
    max_chunk_size=1024,
    overlap=128,
    method="sentences"
)

# Initialize summarizer
summarizer = DocumentSummarizer(
    model_name="facebook/bart-large-cnn",
    max_length=150,
    min_length=50
)

# Generate summary (RAG mode)
result = summarizer.summarize_long_document(
    text=text,
    chunks=chunks,
    method="rag",
    rag_query="main contributions and key findings",
    retrieval_top_k=5,
)
print(result['final_summary'])
```

#### Option 3: Run Example

See a working example:

```bash
python notebooks/example.py
```

#### Testing

Run the test suite to verify installation:

```bash
pytest -q
```

Optional end-to-end demo script:

```bash
python test_pipeline.py
```

## Approach
1. Load and clean long documents (PDF/Text)
2. Split text into overlapping chunks while preserving semantic coherence
3. Generate summaries for individual chunks
4. Combine chunk summaries into a final meta-summary
5. Evaluate summary quality using both automatic and qualitative metrics

### RAG Flow
1. Index chunked document text with lexical scoring
2. Retrieve top-K chunks for a user query
3. Run hierarchical summarization only on retrieved chunks
4. Return summary + retrieval metadata (chunk ids and scores)

## Tech Stack
- Python 3.10+
- HuggingFace Transformers
- BART / T5 / PEGASUS models
- PyPDF2 (PDF extraction)
- spaCy / NLTK (text processing)
- ROUGE, BERTScore (evaluation)
- Streamlit (web interface)

## Evaluation Metrics
- **ROUGE-1 / ROUGE-2 / ROUGE-L**: N-gram overlap metrics
- **BERTScore**: Semantic similarity using embeddings
- **Faithfulness Check**: Detect hallucinations and measure factual consistency
- **Compression Ratio**: Original length vs summary length

## Project Structure

```
long-doc-summarization-nlp/
├── app.py                    # Streamlit web application
├── requirements.txt          # Python dependencies
├── test_pipeline.py         # Test suite
├── data/
│   ├── raw/                 # Raw input documents
│   └── processed/           # Processed outputs
├── notebooks/
│   ├── example.py           # Usage example
│   └── exploration.py       # Experimental code
└── src/
    ├── __init__.py
    ├── loader.py            # PDF/text loading
    ├── chunker.py           # Text chunking
    ├── summarizer.py        # Summarization engine
    └── evaluator.py         # Quality metrics
```

## Supported Models

- **facebook/bart-large-cnn** (Recommended) - Best for general summarization
- **t5-base** / **t5-small** - Flexible text-to-text
- **google/pegasus-cnn_dailymail** - Pre-trained for summarization

## Configuration

Adjust parameters in the Streamlit sidebar or programmatically:

- **Chunk Size**: 256-2048 tokens (default: 1024)
- **Overlap**: 0-512 tokens (default: 128)
- **Chunking Method**: sentences, paragraphs, tokens
- **Summary Length**: min=20-100, max=50-500 words
- **Summarization Method**: hierarchical, concatenate, rag
- **RAG Query**: retrieval objective (e.g., "key findings and conclusions")
- **RAG Top-K**: number of retrieved chunks (default: 5)

## Future Improvements
- Citation-aware summarization
- Extractive + abstractive hybrid models
- Domain-specific fine-tuning (legal, medical)
- Long-context LLM integration (LongT5, LED)
- Multi-language support
- GPU optimization for faster processing

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - See LICENSE file for details

### Used AI to generate all the Readme files