"""
Summarization module using transformer models for long document summarization.
Implements chunk-based and hierarchical summarization strategies.
"""

from typing import List, Optional, Dict
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    pipeline
)
import warnings

warnings.filterwarnings('ignore')


class DocumentSummarizer:
    """
    Summarizes long documents using chunk-based and hierarchical approaches.
    """

    def __init__(
        self,
        model_name: str = "facebook/bart-large-cnn",
        device: Optional[str] = None,
        max_length: int = 150,
        min_length: int = 50
    ):
        """
        Initialize the summarizer with a pre-trained model.

        Args:
            model_name (str): HuggingFace model identifier
            device (str): Device to run on ('cuda', 'cpu', or None for auto)
            max_length (int): Maximum summary length
            min_length (int): Minimum summary length
        """
        self.model_name = model_name
        self.max_length = max_length
        self.min_length = min_length
        
        # Auto-detect device if not specified
        if device is None:
            self.device = 0 if torch.cuda.is_available() else -1
        else:
            self.device = 0 if device == "cuda" else -1
        
        print(f"Loading model: {model_name}...")
        try:
            self.summarizer = pipeline(
                "summarization",
                model=model_name,
                device=self.device,
                model_kwargs={"max_length": 1024}
            )
            print(f"✓ Model loaded successfully on {'GPU' if self.device == 0 else 'CPU'}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def summarize_chunk(
        self,
        text: str,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None
    ) -> str:
        """
        Summarize a single text chunk.

        Args:
            text (str): Input text to summarize
            max_length (int): Override default max length
            min_length (int): Override default min length

        Returns:
            str: Generated summary
        """
        max_len = max_length or self.max_length
        min_len = min_length or self.min_length
        
        # Skip if text is too short
        text_words = len(text.split())
        if text_words < 10:
            return text
        
        # Adjust min_length if text is shorter than min_length
        adjusted_min_len = min(min_len, max(10, text_words // 2))
        adjusted_max_len = min(max_len, text_words)
        
        # Ensure min_length < max_length
        if adjusted_min_len >= adjusted_max_len:
            adjusted_min_len = max(10, adjusted_max_len - 20)
        
        try:
            summary = self.summarizer(
                text,
                max_length=adjusted_max_len,
                min_length=adjusted_min_len,
                do_sample=False,
                truncation=True
            )
            return summary[0]['summary_text']
        except Exception as e:
            print(f"Error summarizing chunk: {e}")
            # Fallback: return first few sentences
            sentences = text.split('.')[:3]
            return '. '.join(sentences).strip() + '.'

    def summarize_chunks(
        self,
        chunks: List[str],
        progress_callback=None
    ) -> List[str]:
        """
        Summarize multiple chunks individually.

        Args:
            chunks (List[str]): List of text chunks
            progress_callback: Optional callback function for progress updates

        Returns:
            List[str]: List of chunk summaries
        """
        summaries = []
        total = len(chunks)
        
        for i, chunk in enumerate(chunks):
            summary = self.summarize_chunk(chunk)
            summaries.append(summary)
            
            if progress_callback:
                progress_callback(i + 1, total)
            else:
                print(f"Summarized chunk {i + 1}/{total}")
        
        return summaries

    def hierarchical_summarize(
        self,
        chunks: List[str],
        final_max_length: Optional[int] = None
    ) -> Dict[str, any]:
        """
        Perform hierarchical summarization:
        1. Summarize individual chunks
        2. Combine chunk summaries
        3. Generate final meta-summary

        Args:
            chunks (List[str]): List of text chunks
            final_max_length (int): Max length for final summary

        Returns:
            Dict containing chunk summaries and final summary
        """
        print(f"\n=== Hierarchical Summarization ===")
        print(f"Processing {len(chunks)} chunks...")
        
        # Step 1: Summarize each chunk
        print("\nStep 1: Summarizing individual chunks...")
        chunk_summaries = self.summarize_chunks(chunks)
        
        # Filter out empty summaries
        chunk_summaries = [s for s in chunk_summaries if s and s.strip()]
        
        if not chunk_summaries:
            print("Warning: No valid chunk summaries generated")
            return {"final_summary": "Unable to generate summary"}
        
        # Step 2: Combine chunk summaries
        print("\nStep 2: Combining chunk summaries...")
        combined_summary = " ".join(chunk_summaries)
        
        # Step 3: Generate final meta-summary
        print("\nStep 3: Generating final meta-summary...")
        final_length = final_max_length or (self.max_length * 2)
        final_summary = self.summarize_chunk(
            combined_summary,
            max_length=final_length,
            min_length=self.min_length
        )
        
        print("✓ Hierarchical summarization complete!")
        
        return final_summary

    def concatenate_and_summarize(
        self,
        chunks: List[str]
    ) -> str:
        """
        Concatenate chunk summaries into final summary.
        
        Args:
            chunks (List[str]): List of text chunks
            
        Returns:
            str: Final concatenated summary
        """
        print(f"\n=== Concatenate Method ===")
        print(f"Processing {len(chunks)} chunks...")
        
        chunk_summaries = self.summarize_chunks(chunks)
        
        # Filter out empty summaries
        chunk_summaries = [s for s in chunk_summaries if s and s.strip()]
        
        concatenated = " ".join(chunk_summaries)
        
        print("✓ Concatenation complete!")
        
        return concatenated

    def summarize_long_document(
        self,
        text: str,
        chunks: List[str],
        method: str = "hierarchical"
    ) -> Dict[str, any]:
        """
        Main method to summarize a long document.

        Args:
            text (str): Original full document text
            chunks (List[str]): Pre-chunked text segments
            method (str): "hierarchical" or "concatenate"

        Returns:
            Dict containing summaries and metadata
        """
        if method == "hierarchical":
            summary = self.hierarchical_summarize(chunks)
            return {
                "final_summary": summary,
                "method": "hierarchical",
                "num_chunks": len(chunks),
                "original_length": len(text.split()),
                "summary_length": len(summary.split()),
                "compression_ratio": len(text.split()) / len(summary.split())
            }
        
        elif method == "concatenate":
            summary = self.concatenate_and_summarize(chunks)
            return {
                "final_summary": summary,
                "method": "concatenate",
                "num_chunks": len(chunks),
                "original_length": len(text.split()),
                "summary_length": len(summary.split())
            }
        
        else:
            raise ValueError(f"Unknown summarization method: {method}")


def quick_summarize(
    text: str,
    model_name: str = "facebook/bart-large-cnn",
    max_length: int = 150
) -> str:
    """
    Quick utility function to summarize text without chunking.

    Args:
        text (str): Input text
        model_name (str): Model to use
        max_length (int): Maximum summary length

    Returns:
        str: Summary text
    """
    summarizer = DocumentSummarizer(model_name=model_name, max_length=max_length)
    return summarizer.summarize_chunk(text)
