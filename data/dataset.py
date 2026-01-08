"""
Dataset manager for processed documents.
Provides easy access to processed papers and their metadata.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import random


class DocumentDataset:
    """Manage and access processed documents."""
    
    def __init__(self, processed_dir="processed"):
        self.base_dir = Path(__file__).parent
        self.processed_dir = self.base_dir / processed_dir
        
        if not self.processed_dir.exists():
            raise FileNotFoundError(f"Processed directory not found: {self.processed_dir}")
        
        self.documents = self._load_documents()
    
    def _load_documents(self) -> List[Dict]:
        """Load all processed documents with metadata."""
        documents = []
        
        for text_file in sorted(self.processed_dir.glob("*.txt")):
            # Skip files that end with _metadata
            if text_file.stem.endswith('_metadata'):
                continue
            
            # Load text
            with open(text_file, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Load metadata
            meta_file = text_file.with_name(f"{text_file.stem}_metadata.json")
            metadata = {}
            
            if meta_file.exists():
                with open(meta_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
            
            documents.append({
                'id': text_file.stem,
                'filename': text_file.name,
                'text': text,
                'metadata': metadata
            })
        
        return documents
    
    def __len__(self):
        """Return number of documents in dataset."""
        return len(self.documents)
    
    def __getitem__(self, idx):
        """Get document by index."""
        return self.documents[idx]
    
    def get_by_id(self, doc_id: str) -> Optional[Dict]:
        """Get document by ID (filename stem)."""
        for doc in self.documents:
            if doc['id'] == doc_id:
                return doc
        return None
    
    def get_by_arxiv_id(self, arxiv_id: str) -> Optional[Dict]:
        """Get document by arXiv ID."""
        for doc in self.documents:
            if doc['metadata'].get('arxiv_id') == arxiv_id:
                return doc
        return None
    
    def list_documents(self) -> List[Dict[str, any]]:
        """List all documents with basic info."""
        return [
            {
                'id': doc['id'],
                'filename': doc['filename'],
                'title': doc['metadata'].get('title', 'Unknown'),
                'arxiv_id': doc['metadata'].get('arxiv_id'),
                'word_count': doc['metadata'].get('word_count', 0)
            }
            for doc in self.documents
        ]
    
    def get_text(self, idx_or_id) -> str:
        """Get document text by index or ID."""
        if isinstance(idx_or_id, int):
            return self.documents[idx_or_id]['text']
        else:
            doc = self.get_by_id(idx_or_id)
            return doc['text'] if doc else None
    
    def get_metadata(self, idx_or_id) -> Dict:
        """Get document metadata by index or ID."""
        if isinstance(idx_or_id, int):
            return self.documents[idx_or_id]['metadata']
        else:
            doc = self.get_by_id(idx_or_id)
            return doc['metadata'] if doc else None
    
    def get_random_document(self) -> Dict:
        """Get a random document from the dataset."""
        return random.choice(self.documents)
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics."""
        if not self.documents:
            return {}
        
        word_counts = [doc['metadata'].get('word_count', 0) for doc in self.documents]
        
        return {
            'total_documents': len(self.documents),
            'total_words': sum(word_counts),
            'avg_words': sum(word_counts) / len(word_counts),
            'min_words': min(word_counts),
            'max_words': max(word_counts),
            'documents_with_arxiv_id': sum(
                1 for doc in self.documents 
                if doc['metadata'].get('arxiv_id')
            )
        }
    
    def filter_by_length(self, min_words: int = 0, max_words: int = float('inf')) -> List[Dict]:
        """Filter documents by word count."""
        return [
            doc for doc in self.documents
            if min_words <= doc['metadata'].get('word_count', 0) <= max_words
        ]
    
    def search_in_titles(self, query: str) -> List[Dict]:
        """Search for documents by title."""
        query = query.lower()
        return [
            doc for doc in self.documents
            if query in doc['metadata'].get('title', '').lower()
        ]
    
    def export_summary(self, output_file: str = "dataset_summary.json"):
        """Export dataset summary to JSON file."""
        summary = {
            'statistics': self.get_statistics(),
            'documents': self.list_documents()
        }
        
        output_path = self.processed_dir / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Summary exported to: {output_path}")
        return output_path


def main():
    """Demo usage of DocumentDataset."""
    try:
        dataset = DocumentDataset()
        
        print("=" * 60)
        print("DOCUMENT DATASET")
        print("=" * 60)
        
        # Statistics
        stats = dataset.get_statistics()
        print(f"\nTotal documents: {stats['total_documents']}")
        print(f"Total words: {stats['total_words']:,}")
        print(f"Average words: {stats['avg_words']:,.0f}")
        print(f"Range: {stats['min_words']:,} - {stats['max_words']:,} words")
        
        # List documents
        print("\n" + "=" * 60)
        print("DOCUMENTS")
        print("=" * 60)
        
        for doc_info in dataset.list_documents():
            print(f"\n[{doc_info['id']}]")
            print(f"  Title: {doc_info['title'][:60]}...")
            print(f"  Words: {doc_info['word_count']:,}")
            if doc_info['arxiv_id']:
                print(f"  arXiv: {doc_info['arxiv_id']}")
        
        # Export summary
        print("\n" + "=" * 60)
        dataset.export_summary()
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run process_pdfs.py first to process the raw PDFs.")


if __name__ == "__main__":
    main()
