"""
Batch processing script for PDF documents.
Processes all PDFs in data/raw and saves cleaned text to data/processed.
"""

import sys
from pathlib import Path
import json
import re
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))
from src.loader import load_pdf, clean_text


class PDFProcessor:
    """Process and manage PDF documents."""
    
    def __init__(self, raw_dir="raw", processed_dir="processed"):
        self.base_dir = Path(__file__).parent
        self.raw_dir = self.base_dir / raw_dir
        self.processed_dir = self.base_dir / processed_dir
        
        # Create processed directory if it doesn't exist
        self.processed_dir.mkdir(exist_ok=True)
        
    def extract_metadata(self, text, filename):
        """Extract metadata from paper text."""
        lines = text.split('\n')[:50]  # Check first 50 lines
        
        metadata = {
            'filename': filename,
            'arxiv_id': self._extract_arxiv_id(filename),
            'title': self._extract_title(lines),
            'abstract': self._extract_abstract(text),
            'processed_date': datetime.now().isoformat(),
            'word_count': len(text.split()),
            'char_count': len(text)
        }
        
        return metadata
    
    def _extract_arxiv_id(self, filename):
        """Extract arXiv ID from filename."""
        match = re.search(r'(\d{4}\.\d{5})', filename)
        return match.group(1) if match else None
    
    def _extract_title(self, lines):
        """Attempt to extract paper title from first lines."""
        for line in lines[:20]:
            line = line.strip()
            # Title is usually one of the first non-empty lines with reasonable length
            if len(line) > 10 and len(line) < 200 and not line.isupper():
                return line
        return "Unknown Title"
    
    def _extract_abstract(self, text):
        """Extract abstract section if present."""
        abstract_match = re.search(
            r'Abstract[:\s]+(.*?)(?:\n\n|\n1\s+Introduction|\nIntroduction)',
            text,
            re.IGNORECASE | re.DOTALL
        )
        
        if abstract_match:
            abstract = abstract_match.group(1).strip()
            # Limit abstract length
            return abstract[:1000] if len(abstract) > 1000 else abstract
        
        return "Abstract not found"
    
    def process_single_pdf(self, pdf_path):
        """Process a single PDF file."""
        print(f"Processing: {pdf_path.name}")
        
        try:
            # Load and clean PDF
            raw_text = load_pdf(str(pdf_path))
            cleaned_text = clean_text(raw_text)
            
            # Extract metadata
            metadata = self.extract_metadata(cleaned_text, pdf_path.name)
            
            # Create output filename (without version suffix)
            base_name = pdf_path.stem.replace('.pdf', '')
            output_name = base_name
            
            # Save cleaned text
            text_file = self.processed_dir / f"{output_name}.txt"
            with open(text_file, 'w', encoding='utf-8') as f:
                f.write(cleaned_text)
            
            # Save metadata
            meta_file = self.processed_dir / f"{output_name}_metadata.json"
            with open(meta_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"  ✓ Saved: {text_file.name} ({metadata['word_count']} words)")
            return True, metadata
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            return False, None
    
    def process_all_pdfs(self):
        """Process all PDF files in raw directory."""
        pdf_files = sorted(self.raw_dir.glob("*.pdf"))
        
        if not pdf_files:
            print("No PDF files found in raw directory.")
            return
        
        print(f"\nFound {len(pdf_files)} PDF files to process\n")
        print("=" * 60)
        
        results = []
        successful = 0
        
        for pdf_file in pdf_files:
            success, metadata = self.process_single_pdf(pdf_file)
            if success:
                successful += 1
                results.append(metadata)
        
        print("=" * 60)
        print(f"\nProcessing complete: {successful}/{len(pdf_files)} successful")
        
        # Save summary
        summary = {
            'processing_date': datetime.now().isoformat(),
            'total_files': len(pdf_files),
            'successful': successful,
            'documents': results
        }
        
        summary_file = self.processed_dir / "processing_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Summary saved to: {summary_file.name}")
        
        return results
    
    def get_processed_files(self):
        """List all processed text files."""
        return sorted(self.processed_dir.glob("*.txt"))
    
    def load_processed_document(self, filename):
        """Load a processed document and its metadata."""
        text_file = self.processed_dir / filename
        
        if not text_file.exists():
            text_file = self.processed_dir / f"{filename}.txt"
        
        if not text_file.exists():
            raise FileNotFoundError(f"Document not found: {filename}")
        
        # Load text
        with open(text_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Load metadata if exists
        meta_file = text_file.with_name(f"{text_file.stem}_metadata.json")
        metadata = None
        
        if meta_file.exists():
            with open(meta_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        
        return text, metadata


def main():
    """Main execution function."""
    processor = PDFProcessor()
    
    print("\n" + "=" * 60)
    print("PDF BATCH PROCESSOR")
    print("=" * 60)
    
    # Process all PDFs
    results = processor.process_all_pdfs()
    
    # Display statistics
    if results:
        print("\n" + "=" * 60)
        print("STATISTICS")
        print("=" * 60)
        
        total_words = sum(r['word_count'] for r in results)
        avg_words = total_words / len(results)
        
        print(f"Total documents: {len(results)}")
        print(f"Total words: {total_words:,}")
        print(f"Average words per document: {avg_words:,.0f}")
        print(f"Shortest document: {min(r['word_count'] for r in results):,} words")
        print(f"Longest document: {max(r['word_count'] for r in results):,} words")


if __name__ == "__main__":
    main()
