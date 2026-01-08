"""
Quick test script - Summarize 1-2 papers to verify everything works.
"""

import sys
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent))

from src.chunker import TextChunker
from src.summarizer import DocumentSummarizer
from src.evaluator import calculate_faithfulness_score
from data.dataset import DocumentDataset


def quick_test(num_docs: int = 2):
    """Quick test on a few documents."""
    print("\n" + "="*60)
    print("QUICK TEST - SUMMARIZING SAMPLE DOCUMENTS")
    print("="*60)
    
    # Load dataset
    print("\nLoading dataset...")
    dataset = DocumentDataset()
    print(f"✓ Loaded {len(dataset)} documents")
    
    # Select shortest documents for quick testing
    docs_by_length = sorted(
        dataset.documents,
        key=lambda d: d['metadata'].get('word_count', 0)
    )
    test_docs = docs_by_length[:num_docs]
    
    print(f"\nSelected {num_docs} shortest documents for testing:")
    for doc in test_docs:
        print(f"  - {doc['id']}: {doc['metadata'].get('word_count', 0):,} words")
    
    # Initialize components
    print("\nInitializing summarizer...")
    summarizer = DocumentSummarizer(
        model_name="facebook/bart-large-cnn",
        max_length=150,
        min_length=50
    )
    
    chunker = TextChunker(chunk_size=1024, overlap=128)
    
    # Process each document
    results = []
    
    for i, doc in enumerate(test_docs, 1):
        print("\n" + "="*60)
        print(f"[{i}/{len(test_docs)}] {doc['id']}")
        print("="*60)
        
        title = doc['metadata'].get('title', 'Unknown')
        print(f"Title: {title[:60]}...")
        print(f"Length: {doc['metadata'].get('word_count', 0):,} words")
        
        # Chunk
        print("\n1. Chunking...")
        start_time = datetime.now()
        chunks = chunker.chunk_text(doc['text'], method="sentences")
        print(f"   ✓ Created {len(chunks)} chunks")
        
        # Summarize
        print("2. Summarizing (hierarchical)...")
        summary = summarizer.hierarchical_summarize(chunks)
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"   ✓ Generated in {duration:.1f}s")
        
        # Metrics
        print("3. Computing metrics...")
        faithfulness = calculate_faithfulness_score(doc['text'], summary)
        compression = len(summary.split()) / len(doc['text'].split())
        
        print(f"   ✓ Compression: {compression:.1%}")
        print(f"   ✓ Faithfulness: {faithfulness:.1%}")
        
        # Display summary
        print(f"\n📝 Summary ({len(summary.split())} words):")
        print("   " + "-"*56)
        print(f"   {summary}")
        print("   " + "-"*56)
        
        results.append({
            'doc_id': doc['id'],
            'title': title,
            'original_words': doc['metadata'].get('word_count', 0),
            'summary_words': len(summary.split()),
            'chunks': len(chunks),
            'compression': compression,
            'faithfulness': faithfulness,
            'time': duration,
            'summary': summary
        })
    
    # Final summary
    print("\n" + "="*60)
    print("QUICK TEST SUMMARY")
    print("="*60)
    
    avg_compression = sum(r['compression'] for r in results) / len(results)
    avg_faithfulness = sum(r['faithfulness'] for r in results) / len(results)
    total_time = sum(r['time'] for r in results)
    
    print(f"\nDocuments processed: {len(results)}")
    print(f"Average compression: {avg_compression:.1%}")
    print(f"Average faithfulness: {avg_faithfulness:.1%}")
    print(f"Total time: {total_time:.1f}s")
    
    print("\n✅ Quick test complete!")
    print("\nTo run full batch summarization:")
    print("  python summarize_dataset.py")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Quick test on sample documents")
    parser.add_argument(
        '--num-docs',
        type=int,
        default=2,
        help='Number of documents to test (default: 2)'
    )
    
    args = parser.parse_args()
    
    quick_test(num_docs=args.num_docs)
