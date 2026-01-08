"""
Test script to verify the long document summarization pipeline.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.chunker import TextChunker, split_long_document
from src.summarizer import DocumentSummarizer
from src.evaluator import SummaryEvaluator, calculate_faithfulness_score


def test_chunker():
    """Test the text chunking module."""
    print("\n" + "="*60)
    print("TESTING CHUNKER MODULE")
    print("="*60)
    
    # Sample long text
    sample_text = """
    Artificial intelligence has revolutionized many aspects of modern life. 
    Machine learning algorithms can now process vast amounts of data to identify 
    patterns and make predictions. Deep learning, a subset of machine learning, 
    uses neural networks with multiple layers to learn complex representations.
    
    Natural language processing has made significant advances in recent years.
    Large language models like GPT and BERT have shown remarkable capabilities
    in understanding and generating human-like text. These models are trained
    on massive datasets containing billions of words from diverse sources.
    
    The future of AI holds both promise and challenges. Ethical considerations
    around AI deployment are becoming increasingly important. Issues like bias,
    fairness, and transparency need careful attention as AI systems become more
    prevalent in critical decision-making processes.
    """ * 5  # Repeat to make it longer
    
    # Test chunking
    chunker = TextChunker(chunk_size=100, overlap=20)
    chunks = chunker.chunk_text(sample_text, method="sentences")
    metadata = chunker.get_chunk_metadata(chunks)
    
    print(f"\n✓ Created {len(chunks)} chunks")
    print(f"✓ Sample chunk preview: {chunks[0][:100]}...")
    
    return True


def test_summarizer():
    """Test the summarization module."""
    print("\n" + "="*60)
    print("TESTING SUMMARIZER MODULE")
    print("="*60)
    
    # Short test text
    test_text = """
    Climate change is one of the most pressing challenges facing humanity today.
    Rising global temperatures are causing ice caps to melt, sea levels to rise,
    and extreme weather events to become more frequent. Scientists worldwide agree
    that human activities, particularly the burning of fossil fuels, are the primary
    cause of recent climate change. Urgent action is needed to reduce greenhouse gas
    emissions and transition to renewable energy sources to mitigate the worst effects
    of climate change and protect future generations.
    """
    
    print("\nInitializing summarizer (this may take a moment)...")
    summarizer = DocumentSummarizer(
        model_name="facebook/bart-large-cnn",
        max_length=50,
        min_length=20
    )
    
    print("\nGenerating summary...")
    summary = summarizer.summarize_chunk(test_text)
    
    print(f"\n✓ Original text ({len(test_text.split())} words):")
    print(f"  {test_text.strip()[:150]}...")
    print(f"\n✓ Generated summary ({len(summary.split())} words):")
    print(f"  {summary}")
    
    return True


def test_evaluator():
    """Test the evaluation module."""
    print("\n" + "="*60)
    print("TESTING EVALUATOR MODULE")
    print("="*60)
    
    # Sample summary and reference
    generated_summary = "Climate change is a major threat. Rising temperatures cause melting ice and extreme weather."
    reference_summary = "Climate change poses significant risks including melting ice caps and increased extreme weather events."
    
    evaluator = SummaryEvaluator(metrics=['rouge1', 'rouge2', 'rougeL'])
    
    print("\n Computing ROUGE scores...")
    results = evaluator.compute_rouge(generated_summary, reference_summary)
    
    print("\n ROUGE Scores:")
    for metric, score in results.items():
        print(f"  {metric}: {score:.4f}")
    
    return True


def test_full_pipeline():
    """Test the complete pipeline."""
    print("\n" + "="*60)
    print("TESTING FULL PIPELINE")
    print("="*60)
    
    # Long sample text
    long_text = """
    The history of computing spans thousands of years, from the abacus to modern 
    supercomputers. The first mechanical computer was designed by Charles Babbage 
    in the 19th century, though it was never fully built during his lifetime.
    
    The 20th century saw rapid advances in computing technology. The invention of 
    the transistor in 1947 revolutionized electronics and paved the way for modern 
    computers. The integrated circuit, developed in the 1950s, further miniaturized 
    electronic components and increased computing power.
    
    Personal computers became widely available in the 1980s, transforming how people 
    work and communicate. The internet, initially developed for military and academic 
    purposes, became publicly accessible in the 1990s and has since connected billions 
    of people worldwide.
    
    Today, computing is ubiquitous. Smartphones put powerful computers in our pockets. 
    Cloud computing allows access to vast resources over the internet. Artificial 
    intelligence and machine learning are pushing the boundaries of what computers 
    can do, from recognizing images to understanding natural language.
    
    The future of computing holds exciting possibilities. Quantum computing promises 
    to solve problems that are intractable for classical computers. Neuromorphic 
    computing aims to mimic the efficiency of biological brains. As technology 
    continues to advance, computing will likely play an even more central role in 
    human society and progress.
    """ * 2
    
    try:
        # Step 1: Chunk the text
        print("\n1. Chunking text...")
        chunks, metadata = split_long_document(
            long_text,
            max_chunk_size=200,
            overlap=50,
            method="sentences"
        )
        print(f" Created {len(chunks)} chunks")
        
        # Step 2: Initialize summarizer
        print("\n2. Initializing summarizer...")
        summarizer = DocumentSummarizer(
            model_name="facebook/bart-large-cnn",
            max_length=80,
            min_length=30
        )
        
        # Step 3: Perform hierarchical summarization
        print("\n3. Performing hierarchical summarization...")
        result = summarizer.hierarchical_summarize(chunks, final_max_length=150)
        
        # Step 4: Display results
        print("\n" + "="*60)
        print("RESULTS")
        print("="*60)
        print(f"\nOriginal length: {len(long_text.split())} words")
        print(f"Number of chunks: {result['num_chunks']}")
        print(f"Final summary length: {len(result['final_summary'].split())} words")
        print(f"\n Final Summary:\n{result['final_summary']}")
        
        # Step 5: Calculate faithfulness
        print("\n4. Evaluating faithfulness...")
        faithfulness = calculate_faithfulness_score(
            result['final_summary'],
            long_text
        )
        print(f"   Faithfulness ratio: {faithfulness['faithfulness_ratio']:.2%}")
        
        print("\n FULL PIPELINE TEST PASSED!")
        return True
        
    except Exception as e:
        print(f"\n Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("LONG DOCUMENT SUMMARIZATION - TEST SUITE")
    print("="*60)
    
    tests = [
        ("Chunker", test_chunker),
        ("Summarizer", test_summarizer),
        ("Evaluator", test_evaluator),
        ("Full Pipeline", test_full_pipeline)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n {test_name} test failed with error: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    for test_name, success in results:
        status = "PASSED" if success else "FAILED"
        print(f"{test_name:.<40} {status}")
    
    all_passed = all(success for _, success in results)
    
    if all_passed:
        print("\n All tests passed successfully!")
    else:
        print("\ Some tests failed. Please check the output above.")
    
    return all_passed


if __name__ == "__main__":
    main()
