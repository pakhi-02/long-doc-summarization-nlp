"""
Simple example demonstrating the long document summarization pipeline.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chunker import split_long_document
from src.summarizer import DocumentSummarizer
from src.loader import load_pdf, clean_text

# Sample long text
sample_document = """
The Industrial Revolution was a period of major industrialization and innovation 
that took place during the late 1700s and early 1800s. The Industrial Revolution 
began in Great Britain and quickly spread throughout the world. The American 
Industrial Revolution commonly referred to as the Second Industrial Revolution, 
started sometime between 1820 and 1870.

This period saw major changes in agriculture, manufacturing, mining, and transport 
that had profound effects on the socioeconomic and cultural conditions of the time. 
Before the Industrial Revolution, goods were produced in small workshops or at home. 
During the Industrial Revolution, production increasingly took place in factories 
using machines powered by water and steam engines.

The textile industry was one of the first to be transformed. The flying shuttle, 
invented by John Kay in 1733, increased the speed of weaving. The spinning jenny, 
invented by James Hargreaves in 1764, could spin multiple threads at once. These 
innovations dramatically increased textile production and reduced costs.

The steam engine, developed by James Watt in the 1770s, was perhaps the most 
important innovation of the Industrial Revolution. Steam power was used not only 
in textile mills but also in transportation, with steam locomotives and steamships 
revolutionizing the movement of goods and people.

The Industrial Revolution had significant social impacts. Urbanization accelerated 
as people moved from rural areas to cities to work in factories. Working conditions 
in early factories were often harsh, with long hours, dangerous conditions, and low 
wages. Child labor was common. These conditions eventually led to the rise of labor 
movements and reforms.

The Industrial Revolution also had environmental consequences. The burning of coal 
for steam power led to air pollution. Factory waste polluted rivers and streams. 
Deforestation occurred as trees were cut down for fuel and to clear land for 
factories and urban development.

Despite its challenges, the Industrial Revolution laid the foundation for the modern 
world. It led to unprecedented economic growth, technological innovation, and improved 
standards of living for many. The principles of mass production, division of labor, 
and mechanization that emerged during this period continue to shape manufacturing 
and industry today.
"""

def main():
    print("="*60)
    print("LONG DOCUMENT SUMMARIZATION - EXAMPLE")
    print("="*60)
    
    # Step 1: Chunk the document
    print("\n1️⃣ Chunking document...")
    chunks, metadata = split_long_document(
        text=sample_document,
        max_chunk_size=150,  # tokens
        overlap=30,
        method="sentences"
    )
    print(f"   Created {len(chunks)} chunks")
    for i, meta in enumerate(metadata[:3]):  # Show first 3
        print(f"   Chunk {i}: {meta['word_count']} words")
    
    # Step 2: Initialize summarizer
    print("\n2️⃣ Initializing summarizer...")
    summarizer = DocumentSummarizer(
        model_name="facebook/bart-large-cnn",
        max_length=100,
        min_length=40
    )
    
    # Step 3: Generate summary
    print("\n3️⃣ Generating hierarchical summary...")
    result = summarizer.hierarchical_summarize(chunks)
    
    # Step 4: Display results
    print("\n" + "="*60)
    print("📊 RESULTS")
    print("="*60)
    
    original_length = len(sample_document.split())
    summary_length = len(result['final_summary'].split())
    
    print(f"\n📄 Original Document: {original_length} words")
    print(f"📦 Number of Chunks: {result['num_chunks']}")
    print(f"📝 Summary Length: {summary_length} words")
    print(f"📉 Compression Ratio: {original_length / summary_length:.2f}x")
    
    print("\n" + "="*60)
    print("✨ FINAL SUMMARY")
    print("="*60)
    print(f"\n{result['final_summary']}\n")
    
    # Optional: Show chunk summaries
    print("\n" + "="*60)
    print("📑 CHUNK SUMMARIES")
    print("="*60)
    for i, chunk_summary in enumerate(result['chunk_summaries']):
        print(f"\nChunk {i+1}:")
        print(f"  {chunk_summary}")
    
    print("\n✅ Example completed successfully!")

if __name__ == "__main__":
    main()
