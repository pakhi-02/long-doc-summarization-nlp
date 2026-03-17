"""
Summarization module using transformer models for long document summarization.
Implements chunk-based and hierarchical summarization strategies.
"""

from typing import List, Optional, Dict, Any
import os
import torch
import warnings

from src.retriever import ChunkRetriever

warnings.filterwarnings("ignore")


class DocumentSummarizer:
    """
    Summarizes long documents using chunk-based and hierarchical approaches.
    """

    def __init__(
        self,
        model_name: str = "facebook/bart-large-cnn",
        device: Optional[str] = None,
        max_length: int = 150,
        min_length: int = 50,
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
            self.summarizer = self._build_pipeline()
            try:
                self.tokenizer = self._build_tokenizer()
                self.max_input_tokens = self._resolve_max_input_tokens()
            except Exception:
                self.tokenizer = None
                self.max_input_tokens = 1024
            print(f"✓ Model loaded successfully on {'GPU' if self.device == 0 else 'CPU'}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def _build_pipeline(self):
        os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
        from transformers import pipeline

        return pipeline(
            "summarization",
            model=self.model_name,
            device=self.device,
            model_kwargs={"max_length": 1024},
        )

    def _build_tokenizer(self):
        os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(self.model_name)

    def _resolve_max_input_tokens(self) -> int:
        if self.tokenizer is None:
            return 1024
        model_max_length = getattr(self.tokenizer, "model_max_length", None)
        if isinstance(model_max_length, int) and 0 < model_max_length < 100000:
            return model_max_length
        return 1024

    def _truncate_to_model_input(self, text: str) -> str:
        if self.tokenizer is None:
            return text
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_input_tokens,
            return_tensors=None,
        )
        token_ids = encoded.get("input_ids", [])
        if not token_ids:
            return text

        if isinstance(token_ids[0], list):
            token_ids = token_ids[0]

        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def summarize_chunk(
        self, text: str, max_length: Optional[int] = None, min_length: Optional[int] = None
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

        safe_text = self._truncate_to_model_input(text)

        # Skip if text is too short
        text_words = len(safe_text.split())
        if text_words < 10:
            return safe_text

        # Adjust min_length if text is shorter than min_length
        adjusted_min_len = min(min_len, max(10, text_words // 2))
        adjusted_max_len = min(max_len, text_words)

        # Ensure min_length < max_length
        if adjusted_min_len >= adjusted_max_len:
            adjusted_min_len = max(10, adjusted_max_len - 20)

        try:
            summary = self.summarizer(
                safe_text,
                max_length=adjusted_max_len,
                min_length=adjusted_min_len,
                do_sample=False,
                truncation=False,
            )
            return summary[0]["summary_text"]
        except Exception as e:
            print(f"Error summarizing chunk: {e}")
            # Fallback: return first few sentences
            sentences = safe_text.split(".")[:3]
            return ". ".join(sentences).strip() + "."

    def summarize_chunks(self, chunks: List[str], progress_callback=None) -> List[str]:
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
        self, chunks: List[str], final_max_length: Optional[int] = None
    ) -> Dict[str, Any]:
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
        print("\n=== Hierarchical Summarization ===")
        print(f"Processing {len(chunks)} chunks...")

        # Step 1: Summarize each chunk
        print("\nStep 1: Summarizing individual chunks...")
        chunk_summaries = self.summarize_chunks(chunks)

        # Filter out empty summaries
        chunk_summaries = [s for s in chunk_summaries if s and s.strip()]

        if not chunk_summaries:
            print("Warning: No valid chunk summaries generated")
            return {
                "final_summary": "Unable to generate summary",
                "chunk_summaries": [],
                "num_chunks": len(chunks),
                "combined_summary": "",
            }

        # Step 2: Combine chunk summaries
        print("\nStep 2: Combining chunk summaries...")
        combined_summary = " ".join(chunk_summaries)

        # Step 3: Generate final meta-summary
        print("\nStep 3: Generating final meta-summary...")
        final_length = final_max_length or (self.max_length * 2)
        final_summary = self.summarize_chunk(
            combined_summary, max_length=final_length, min_length=self.min_length
        )

        print("✓ Hierarchical summarization complete!")

        return {
            "final_summary": final_summary,
            "chunk_summaries": chunk_summaries,
            "num_chunks": len(chunk_summaries),
            "combined_summary": combined_summary,
        }

    def concatenate_and_summarize(self, chunks: List[str]) -> Dict[str, Any]:
        """
        Concatenate chunk summaries into final summary.

        Args:
            chunks (List[str]): List of text chunks

        Returns:
            str: Final concatenated summary
        """
        print("\n=== Concatenate Method ===")
        print(f"Processing {len(chunks)} chunks...")

        chunk_summaries = self.summarize_chunks(chunks)

        # Filter out empty summaries
        chunk_summaries = [s for s in chunk_summaries if s and s.strip()]

        concatenated = " ".join(chunk_summaries)

        print("✓ Concatenation complete!")

        return {
            "final_summary": concatenated,
            "chunk_summaries": chunk_summaries,
            "num_chunks": len(chunk_summaries),
        }

    def rag_summarize(
        self,
        chunks: List[str],
        query: Optional[str] = None,
        retrieval_top_k: int = 5,
        final_max_length: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Retrieve the most relevant chunks and summarize them hierarchically.

        Args:
            chunks (List[str]): List of document chunks.
            query (Optional[str]): Retrieval query. If not provided, a default objective is used.
            retrieval_top_k (int): Number of chunks to retrieve.
            final_max_length (Optional[int]): Max length for final summary.

        Returns:
            Dict[str, Any]: Retrieved chunk metadata and final summary.
        """
        print("\n=== RAG Summarization ===")

        retrieval_query = query or "main contributions key findings methodology conclusions"
        retriever = ChunkRetriever(chunks)
        retrieved = retriever.retrieve(retrieval_query, top_k=retrieval_top_k)

        if not retrieved:
            print("No relevant chunks retrieved; falling back to hierarchical summarization.")
            fallback = self.hierarchical_summarize(chunks, final_max_length=final_max_length)
            fallback.update(
                {
                    "retrieval_query": retrieval_query,
                    "retrieved_chunk_ids": [],
                    "retrieved_scores": [],
                }
            )
            return fallback

        retrieved_chunk_ids = [item.chunk_id for item in retrieved]
        retrieved_scores = [item.score for item in retrieved]
        retrieved_chunks = [item.text for item in retrieved]

        print(f"Retrieved {len(retrieved_chunks)} chunks for query: {retrieval_query}")
        summary_result = self.hierarchical_summarize(
            retrieved_chunks,
            final_max_length=final_max_length,
        )
        summary_result.update(
            {
                "retrieval_query": retrieval_query,
                "retrieved_chunk_ids": retrieved_chunk_ids,
                "retrieved_scores": retrieved_scores,
            }
        )
        return summary_result

    def summarize_long_document(
        self,
        text: str,
        chunks: List[str],
        method: str = "hierarchical",
        rag_query: Optional[str] = None,
        retrieval_top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Main method to summarize a long document.

        Args:
            text (str): Original full document text
            chunks (List[str]): Pre-chunked text segments
            method (str): "hierarchical", "concatenate", or "rag"
            rag_query (Optional[str]): Optional query used when method is "rag"
            retrieval_top_k (int): Number of chunks to retrieve when method is "rag"

        Returns:
            Dict containing summaries and metadata
        """
        if method == "hierarchical":
            summary_result = self.hierarchical_summarize(chunks)
            final_summary = summary_result["final_summary"]
            summary_length = max(1, len(final_summary.split()))
            return {
                "final_summary": final_summary,
                "chunk_summaries": summary_result.get("chunk_summaries", []),
                "method": "hierarchical",
                "num_chunks": summary_result.get("num_chunks", len(chunks)),
                "original_length": len(text.split()),
                "summary_length": summary_length,
                "compression_ratio": len(text.split()) / summary_length,
            }

        elif method == "concatenate":
            summary_result = self.concatenate_and_summarize(chunks)
            final_summary = summary_result["final_summary"]
            summary_length = max(1, len(final_summary.split()))
            return {
                "final_summary": final_summary,
                "chunk_summaries": summary_result.get("chunk_summaries", []),
                "method": "concatenate",
                "num_chunks": summary_result.get("num_chunks", len(chunks)),
                "original_length": len(text.split()),
                "summary_length": summary_length,
                "compression_ratio": len(text.split()) / summary_length,
            }

        elif method == "rag":
            summary_result = self.rag_summarize(
                chunks,
                query=rag_query,
                retrieval_top_k=retrieval_top_k,
                final_max_length=self.max_length * 2,
            )
            final_summary = summary_result["final_summary"]
            summary_length = max(1, len(final_summary.split()))
            return {
                "final_summary": final_summary,
                "chunk_summaries": summary_result.get("chunk_summaries", []),
                "method": "rag",
                "num_chunks": summary_result.get("num_chunks", 0),
                "original_length": len(text.split()),
                "summary_length": summary_length,
                "compression_ratio": len(text.split()) / summary_length,
                "retrieval_query": summary_result.get("retrieval_query", ""),
                "retrieved_chunk_ids": summary_result.get("retrieved_chunk_ids", []),
                "retrieved_scores": summary_result.get("retrieved_scores", []),
            }

        else:
            raise ValueError(f"Unknown summarization method: {method}")


def quick_summarize(
    text: str, model_name: str = "facebook/bart-large-cnn", max_length: int = 150
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
