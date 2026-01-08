# Long Document Text Summarizer

This project implements a long-document text summarization system designed for domains such as legal, medical, and research documents. The system handles documents exceeding standard transformer context limits by applying chunking strategies and hierarchical summarization to generate accurate and faithful summaries.

## Problem Statement
Transformer-based summarization models struggle with long documents due to context length limitations, often resulting in information loss or hallucinations. This project addresses these challenges using chunk-based and hierarchical summarization techniques.

## Approach
1. Load and clean long documents (PDF/Text)
2. Split text into overlapping chunks while preserving semantic coherence
3. Generate summaries for individual chunks
4. Combine chunk summaries into a final meta-summary
5. Evaluate summary quality using both automatic and qualitative metrics

## Tech Stack
- Python
- HuggingFace Transformers
- Long-T5 / PEGASUS
- PyPDF2
- spaCy / NLTK
- ROUGE, BERTScore
- Streamlit (for demo)
