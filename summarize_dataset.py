"""
Batch summarization script for the processed dataset.
Summarizes all papers and saves results with metrics.
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List

# Add paths
sys.path.append(str(Path(__file__).parent))

from src.chunker import TextChunker
from src.summarizer import DocumentSummarizer
from src.evaluator import calculate_faithfulness_score
from data.dataset import DocumentDataset


class DatasetSummarizer:
    """Summarize all documents in the dataset."""
    
    def __init__(
        self,
        model_name: str = "facebook/bart-large-cnn",
        chunk_size: int = 1024,
        overlap: int = 128,
        max_length: int = 150,
        min_length: int = 50
    ):
        """Initialize the dataset summarizer."""
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.max_length = max_length
        self.min_length = min_length
        
        # Output directory
        self.output_dir = Path(__file__).parent / "results"
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        print("Loading dataset...")
        self.dataset = DocumentDataset()
        
        print(f"Initializing summarizer with {model_name}...")
        self.summarizer = DocumentSummarizer(
            model_name=model_name,
            max_length=max_length,
            min_length=min_length
        )
        
        self.chunker = TextChunker(
            chunk_size=chunk_size,
            overlap=overlap
        )
        
    def summarize_document(
        self,
        doc: Dict,
        method: str = "hierarchical"
    ) -> Dict:
        """Summarize a single document."""
        doc_id = doc['id']
        text = doc['text']
        metadata = doc['metadata']
        
        print(f"\n{'='*60}")
        print(f"Summarizing: {doc_id}")
        print(f"Title: {metadata.get('title', 'Unknown')[:60]}...")
        print(f"Length: {metadata.get('word_count', 0):,} words")
        print(f"{'='*60}")
        
        # Chunk the document
        print("1. Chunking document...")
        chunks = self.chunker.chunk_text(text, method="sentences")
        print(f"   ✓ Created {len(chunks)} chunks")
        
        # Summarize
        print(f"2. Generating {method} summary...")
        start_time = datetime.now()
        
        if method == "hierarchical":
            summary = self.summarizer.hierarchical_summarize(chunks)
        else:
            summary = self.summarizer.concatenate_and_summarize(chunks)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"   ✓ Summary generated in {duration:.1f}s")
        
        # Calculate metrics
        print("3. Computing metrics...")
        faithfulness = calculate_faithfulness_score(text, summary)
        compression_ratio = len(summary.split()) / len(text.split())
        
        result = {
            'doc_id': doc_id,
            'metadata': metadata,
            'original_length': len(text.split()),
            'summary_length': len(summary.split()),
            'num_chunks': len(chunks),
            'compression_ratio': compression_ratio,
            'faithfulness_score': faithfulness,
            'processing_time': duration,
            'summary': summary,
            'method': method,
            'model': self.model_name,
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"   ✓ Compression: {compression_ratio:.1%}")
        print(f"   ✓ Faithfulness: {faithfulness:.1%}")
        
        return result
    
    def summarize_all(
        self,
        method: str = "hierarchical",
        save_individual: bool = True
    ) -> List[Dict]:
        """Summarize all documents in the dataset."""
        print("\n" + "="*60)
        print("BATCH SUMMARIZATION")
        print("="*60)
        print(f"Dataset: {len(self.dataset)} documents")
        print(f"Method: {method}")
        print(f"Model: {self.model_name}")
        print("="*60)
        
        results = []
        
        for i, doc in enumerate(self.dataset.documents, 1):
            print(f"\n[{i}/{len(self.dataset)}]")
            
            try:
                result = self.summarize_document(doc, method=method)
                results.append(result)
                
                # Save individual result
                if save_individual:
                    individual_file = self.output_dir / f"{doc['id']}_summary.json"
                    with open(individual_file, 'w', encoding='utf-8') as f:
                        json.dump(result, f, indent=2)
                    print(f"   ✓ Saved to {individual_file.name}")
                
            except Exception as e:
                print(f"   ✗ Error: {e}")
                results.append({
                    'doc_id': doc['id'],
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
        
        # Save batch results
        print("\n" + "="*60)
        print("Saving batch results...")
        
        batch_file = self.output_dir / f"batch_results_{method}.json"
        with open(batch_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        print(f"✓ Batch results saved to {batch_file.name}")
        
        # Generate summary report
        self.generate_report(results, method)
        
        return results
    
    def generate_report(self, results: List[Dict], method: str):
        """Generate a summary report."""
        print("\n" + "="*60)
        print("SUMMARY REPORT")
        print("="*60)
        
        successful = [r for r in results if 'summary' in r]
        failed = [r for r in results if 'error' in r]
        
        print(f"\nSuccessful: {len(successful)}/{len(results)}")
        print(f"Failed: {len(failed)}/{len(results)}")
        
        if not successful:
            return
        
        # Calculate statistics
        total_original = sum(r['original_length'] for r in successful)
        total_summary = sum(r['summary_length'] for r in successful)
        avg_compression = sum(r['compression_ratio'] for r in successful) / len(successful)
        avg_faithfulness = sum(r['faithfulness_score'] for r in successful) / len(successful)
        avg_time = sum(r['processing_time'] for r in successful) / len(successful)
        total_time = sum(r['processing_time'] for r in successful)
        
        print(f"\n📊 Statistics:")
        print(f"   Total original words: {total_original:,}")
        print(f"   Total summary words: {total_summary:,}")
        print(f"   Average compression: {avg_compression:.1%}")
        print(f"   Average faithfulness: {avg_faithfulness:.1%}")
        print(f"   Average time per doc: {avg_time:.1f}s")
        print(f"   Total processing time: {total_time:.1f}s ({total_time/60:.1f} min)")
        
        # Save report
        report = {
            'method': method,
            'model': self.model_name,
            'timestamp': datetime.now().isoformat(),
            'statistics': {
                'total_documents': len(results),
                'successful': len(successful),
                'failed': len(failed),
                'total_original_words': total_original,
                'total_summary_words': total_summary,
                'avg_compression_ratio': avg_compression,
                'avg_faithfulness_score': avg_faithfulness,
                'avg_processing_time': avg_time,
                'total_processing_time': total_time
            },
            'documents': [
                {
                    'doc_id': r['doc_id'],
                    'title': r['metadata'].get('title', 'Unknown')[:80],
                    'original_words': r['original_length'],
                    'summary_words': r['summary_length'],
                    'compression': r['compression_ratio'],
                    'faithfulness': r['faithfulness_score'],
                    'time': r['processing_time']
                }
                for r in successful
            ]
        }
        
        report_file = self.output_dir / f"summary_report_{method}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n✓ Report saved to {report_file.name}")
        
        # Print top summaries
        print(f"\n📝 Sample Summaries:")
        for i, r in enumerate(successful[:3], 1):
            print(f"\n{i}. {r['metadata'].get('title', 'Unknown')[:60]}...")
            print(f"   {r['summary'][:200]}...")


def main():
    """Main execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Summarize dataset documents")
    parser.add_argument(
        '--method',
        choices=['hierarchical', 'concatenate'],
        default='hierarchical',
        help='Summarization method'
    )
    parser.add_argument(
        '--model',
        default='facebook/bart-large-cnn',
        help='Model to use for summarization'
    )
    parser.add_argument(
        '--max-length',
        type=int,
        default=150,
        help='Maximum summary length'
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=1024,
        help='Chunk size for splitting'
    )
    
    args = parser.parse_args()
    
    # Initialize and run
    summarizer = DatasetSummarizer(
        model_name=args.model,
        chunk_size=args.chunk_size,
        max_length=args.max_length
    )
    
    results = summarizer.summarize_all(method=args.method)
    
    print("\n" + "="*60)
    print("✅ BATCH SUMMARIZATION COMPLETE!")
    print("="*60)
    print(f"Results saved in: {summarizer.output_dir}")


if __name__ == "__main__":
    main()
