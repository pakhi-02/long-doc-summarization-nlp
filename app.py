"""
Streamlit web application for Long Document Summarization.
"""

import streamlit as st
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.loader import load_pdf, clean_text
from src.chunker import TextChunker, split_long_document
from src.summarizer import DocumentSummarizer
from src.evaluator import SummaryEvaluator, calculate_faithfulness_score, print_evaluation_report


# Page configuration
st.set_page_config(
    page_title="Long Document Summarizer",
    page_icon="📄",
    layout="wide"
)

# Title and description
st.title("📄 Long Document Text Summarizer")
st.markdown("""
This application summarizes long documents (PDF or text) using transformer-based models.
It handles documents that exceed standard context limits through intelligent chunking and hierarchical summarization.
""")

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")

# Model selection
model_options = {
    "BART (Recommended)": "facebook/bart-large-cnn",
    "T5 Base": "t5-base",
    "T5 Small (Faster)": "t5-small",
    "Pegasus": "google/pegasus-cnn_dailymail"
}
selected_model = st.sidebar.selectbox(
    "Select Model",
    options=list(model_options.keys()),
    index=0
)
model_name = model_options[selected_model]

# Chunking parameters
st.sidebar.subheader("Chunking Settings")
chunk_size = st.sidebar.slider("Chunk Size (tokens)", 256, 2048, 1024, 128)
overlap = st.sidebar.slider("Overlap Size", 0, 512, 128, 64)
chunk_method = st.sidebar.selectbox(
    "Chunking Method",
    ["sentences", "paragraphs", "tokens"]
)

# Summarization parameters
st.sidebar.subheader("Summarization Settings")
max_summary_length = st.sidebar.slider("Max Summary Length", 50, 500, 150, 25)
min_summary_length = st.sidebar.slider("Min Summary Length", 20, 100, 50, 10)
summarization_method = st.sidebar.selectbox(
    "Method",
    ["hierarchical", "concatenate"]
)

# Main content
tab1, tab2, tab3 = st.tabs(["📤 Upload & Summarize", "📊 Evaluation", "ℹ️ About"])

with tab1:
    st.header("Document Input")
    
    # Input method selection
    input_method = st.radio(
        "Choose input method:",
        ["Upload PDF", "Paste Text"]
    )
    
    document_text = None
    
    if input_method == "Upload PDF":
        uploaded_file = st.file_uploader("Upload a PDF file", type=['pdf'])
        
        if uploaded_file:
            with st.spinner("📖 Reading PDF..."):
                try:
                    # Save uploaded file temporarily
                    temp_path = f"/tmp/{uploaded_file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Load and clean
                    raw_text = load_pdf(temp_path)
                    document_text = clean_text(raw_text)
                    
                    st.success(f"✅ PDF loaded successfully! ({len(document_text.split())} words)")
                    
                except Exception as e:
                    st.error(f"Error loading PDF: {e}")
    
    else:  # Paste Text
        document_text = st.text_area(
            "Paste your document text here:",
            height=300,
            placeholder="Enter or paste your long document here..."
        )
    
    # Show document preview
    if document_text:
        with st.expander("📄 Document Preview"):
            st.text(document_text[:1000] + "..." if len(document_text) > 1000 else document_text)
            st.metric("Word Count", len(document_text.split()))
    
    # Summarize button
    if document_text and st.button("🚀 Generate Summary", type="primary"):
        
        # Step 1: Chunking
        with st.spinner("✂️ Chunking document..."):
            chunker = TextChunker(chunk_size=chunk_size, overlap=overlap)
            chunks = chunker.chunk_text(document_text, method=chunk_method)
            chunk_metadata = chunker.get_chunk_metadata(chunks)
            
            st.success(f"✅ Created {len(chunks)} chunks")
            
            # Show chunks info
            with st.expander(f"View Chunks ({len(chunks)})"):
                for meta in chunk_metadata:
                    st.markdown(f"**Chunk {meta['chunk_id']}** - {meta['word_count']} words")
                    st.text(meta['preview'])
                    st.divider()
        
        # Step 2: Summarization
        with st.spinner(f"🤖 Summarizing with {selected_model}..."):
            try:
                summarizer = DocumentSummarizer(
                    model_name=model_name,
                    max_length=max_summary_length,
                    min_length=min_summary_length
                )
                
                # Progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                result = summarizer.summarize_long_document(
                    text=document_text,
                    chunks=chunks,
                    method=summarization_method
                )
                
                progress_bar.progress(100)
                status_text.success("✅ Summarization complete!")
                
                # Store in session state
                st.session_state['summary_result'] = result
                st.session_state['document_text'] = document_text
                
            except Exception as e:
                st.error(f"Error during summarization: {e}")
                st.stop()
        
        # Display results
        st.header("📝 Summary Results")
        
        # Final summary
        st.subheader("Final Summary")
        st.info(result['final_summary'])
        
        # Statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Original Length", f"{result['original_length']} words")
        with col2:
            st.metric("Summary Length", f"{result['summary_length']} words")
        with col3:
            st.metric("Compression Ratio", f"{result.get('compression_ratio', 0):.2f}x")
        
        # Chunk summaries
        with st.expander("View Individual Chunk Summaries"):
            for i, chunk_summary in enumerate(result['chunk_summaries']):
                st.markdown(f"**Chunk {i+1} Summary:**")
                st.write(chunk_summary)
                st.divider()

with tab2:
    st.header("📊 Evaluation Metrics")
    
    if 'summary_result' in st.session_state:
        result = st.session_state['summary_result']
        document_text = st.session_state['document_text']
        
        st.subheader("Automatic Metrics")
        
        # Reference summary input
        reference_summary = st.text_area(
            "Paste reference/gold summary (optional):",
            height=150,
            help="If you have a human-written reference summary, paste it here for ROUGE/BERTScore evaluation"
        )
        
        if reference_summary and st.button("Calculate ROUGE & BERTScore"):
            with st.spinner("Computing evaluation metrics..."):
                evaluator = SummaryEvaluator(metrics=['rouge1', 'rouge2', 'rougeL', 'bertscore'])
                eval_results = evaluator.evaluate(
                    summary=result['final_summary'],
                    reference=reference_summary,
                    include_bertscore=True
                )
                
                # Display metrics
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### ROUGE Scores")
                    for key, value in eval_results.items():
                        if 'rouge' in key.lower():
                            st.metric(key.replace('_', ' ').title(), f"{value:.4f}")
                
                with col2:
                    st.markdown("### BERTScore")
                    for key, value in eval_results.items():
                        if 'bertscore' in key.lower():
                            st.metric(key.replace('_', ' ').title(), f"{value:.4f}")
        
        # Faithfulness check
        st.subheader("Faithfulness Analysis")
        if st.button("Check Faithfulness"):
            with st.spinner("Analyzing faithfulness..."):
                faithfulness = calculate_faithfulness_score(
                    summary=result['final_summary'],
                    original_text=document_text
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Faithfulness Ratio", f"{faithfulness['faithfulness_ratio']:.2%}")
                with col2:
                    st.metric("Potential Hallucination Rate", f"{faithfulness['hallucination_ratio']:.2%}")
                
                st.info(faithfulness['note'])
    else:
        st.info("Generate a summary first to see evaluation metrics.")

with tab3:
    st.header("About This Application")
    
    st.markdown("""
    ### 🎯 Purpose
    This application addresses the challenge of summarizing long documents that exceed the context 
    limits of standard transformer models (typically 512-1024 tokens).
    
    ### 🔧 How It Works
    1. **Document Loading**: Extracts text from PDFs or accepts pasted text
    2. **Intelligent Chunking**: Splits documents into manageable chunks with overlap
    3. **Chunk Summarization**: Summarizes each chunk individually
    4. **Hierarchical Aggregation**: Combines chunk summaries into a coherent final summary
    5. **Evaluation**: Measures quality using ROUGE, BERTScore, and faithfulness metrics
    
    ### 📊 Supported Models
    - **BART**: Best for general summarization (CNN/DailyMail)
    - **T5**: Flexible text-to-text transformer
    - **PEGASUS**: Specifically pre-trained for summarization
    
    ### 🎓 Use Cases
    - Legal document analysis
    - Medical report summarization
    - Research paper digests
    - Long-form content condensation
    
    ### ⚠️ Limitations
    - Model quality depends on pre-training data
    - May produce hallucinations for very technical content
    - Processing time increases with document length
    
    ### 🚀 Future Improvements
    - Support for more models (LongT5, LED)
    - Citation-aware summarization
    - Multi-language support
    - Fine-tuning on domain-specific data
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 💡 Tips")
st.sidebar.markdown("""
- Use **hierarchical** method for very long documents
- Increase **overlap** to preserve context
- **Sentence-based** chunking works best for most cases
- Larger models produce better quality but are slower
""")
