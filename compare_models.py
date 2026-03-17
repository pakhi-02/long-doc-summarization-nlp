"""
Compare different summarization models and methods.
Tests multiple configurations and generates comparison visualizations.
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(str(Path(__file__).parent))

from src.chunker import TextChunker
from src.summarizer import DocumentSummarizer
from src.evaluator import calculate_faithfulness_score
from data.dataset import DocumentDataset


class ModelComparison:
    """Compare different models and methods."""
    
    def __init__(self, output_dir: str = "comparison_results"):
        """Initialize comparison framework."""
        self.output_dir = Path(__file__).parent / output_dir
        self.output_dir.mkdir(exist_ok=True)
        
        self.dataset = DocumentDataset()
        self.chunker = TextChunker(chunk_size=1024, overlap=128)
        
    def compare_models(
        self,
        models: List[str] = None,
        num_docs: int = 3,
        methods: List[str] = None
    ) -> Dict:
        """Compare multiple models on sample documents."""
        
        if models is None:
            models = [
                "facebook/bart-large-cnn",
                "t5-base",
                "google/pegasus-cnn_dailymail"
            ]
        
        if methods is None:
            methods = ["hierarchical", "concatenate"]
        
        # Select test documents (shortest for speed)
        test_docs = sorted(
            self.dataset.documents,
            key=lambda d: d['metadata'].get('word_count', 0)
        )[:num_docs]
        
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)
        print(f"Models: {len(models)}")
        print(f"Methods: {len(methods)}")
        print(f"Documents: {num_docs}")
        print("="*60)
        
        all_results = []
        
        for model_name in models:
            print(f"\n{'='*60}")
            print(f"Testing model: {model_name}")
            print(f"{'='*60}")
            
            try:
                summarizer = DocumentSummarizer(
                    model_name=model_name,
                    max_length=150,
                    min_length=50
                )
                
                for method in methods:
                    print(f"\nMethod: {method}")
                    
                    for i, doc in enumerate(test_docs, 1):
                        doc_id = doc['id']
                        print(f"  [{i}/{num_docs}] {doc_id}...", end=" ")
                        
                        try:
                            # Chunk
                            chunks = self.chunker.chunk_text(doc['text'], method="sentences")
                            
                            # Summarize
                            start = datetime.now()
                            if method == "hierarchical":
                                summary_result = summarizer.hierarchical_summarize(chunks)
                            else:
                                summary_result = summarizer.concatenate_and_summarize(chunks)
                            summary = summary_result['final_summary']
                            duration = (datetime.now() - start).total_seconds()
                            
                            # Metrics
                            faithfulness = calculate_faithfulness_score(doc['text'], summary)
                            compression = len(summary.split()) / len(doc['text'].split())
                            
                            result = {
                                'model': model_name,
                                'method': method,
                                'doc_id': doc_id,
                                'original_words': len(doc['text'].split()),
                                'summary_words': len(summary.split()),
                                'compression_ratio': compression,
                                'faithfulness_score': faithfulness,
                                'processing_time': duration,
                                'summary': summary
                            }
                            
                            all_results.append(result)
                            print(f"✓ ({duration:.1f}s)")
                            
                        except Exception as e:
                            print(f"✗ Error: {e}")
                
            except Exception as e:
                print(f"  ✗ Failed to load model: {e}")
        
        # Save results
        results_file = self.output_dir / "model_comparison.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\n✓ Results saved to {results_file}")
        
        # Generate visualizations
        if all_results:
            self.visualize_comparison(all_results)
        
        # Print summary
        self.print_comparison_summary(all_results, models, methods)
        
        return all_results
    
    def visualize_comparison(self, results: List[Dict]):
        """Create comparison visualizations."""
        print("\nGenerating visualizations...")
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (14, 10)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Model Comparison Results', fontsize=16, fontweight='bold')
        
        # Extract data
        models = [r['model'].split('/')[-1] for r in results]
        methods = [r['method'] for r in results]
        compressions = [r['compression_ratio'] for r in results]
        faithfulness = [r['faithfulness_score'] for r in results]
        times = [r['processing_time'] for r in results]
        
        # 1. Compression ratio by model
        ax1 = axes[0, 0]
        model_compressions = {}
        for r in results:
            model = r['model'].split('/')[-1]
            if model not in model_compressions:
                model_compressions[model] = []
            model_compressions[model].append(r['compression_ratio'])
        
        ax1.bar(model_compressions.keys(), 
                [sum(v)/len(v) for v in model_compressions.values()])
        ax1.set_title('Average Compression Ratio by Model')
        ax1.set_ylabel('Compression Ratio')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Faithfulness by model
        ax2 = axes[0, 1]
        model_faithfulness = {}
        for r in results:
            model = r['model'].split('/')[-1]
            if model not in model_faithfulness:
                model_faithfulness[model] = []
            model_faithfulness[model].append(r['faithfulness_score'])
        
        ax2.bar(model_faithfulness.keys(),
                [sum(v)/len(v) for v in model_faithfulness.values()],
                color='green', alpha=0.7)
        ax2.set_title('Average Faithfulness Score by Model')
        ax2.set_ylabel('Faithfulness Score')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Processing time by model
        ax3 = axes[1, 0]
        model_times = {}
        for r in results:
            model = r['model'].split('/')[-1]
            if model not in model_times:
                model_times[model] = []
            model_times[model].append(r['processing_time'])
        
        ax3.bar(model_times.keys(),
                [sum(v)/len(v) for v in model_times.values()],
                color='orange', alpha=0.7)
        ax3.set_title('Average Processing Time by Model')
        ax3.set_ylabel('Time (seconds)')
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. Method comparison
        ax4 = axes[1, 1]
        method_metrics = {}
        for r in results:
            method = r['method']
            if method not in method_metrics:
                method_metrics[method] = {'compression': [], 'faithfulness': []}
            method_metrics[method]['compression'].append(r['compression_ratio'])
            method_metrics[method]['faithfulness'].append(r['faithfulness_score'])
        
        x = range(len(method_metrics))
        width = 0.35
        
        compressions_avg = [sum(v['compression'])/len(v['compression']) 
                           for v in method_metrics.values()]
        faithfulness_avg = [sum(v['faithfulness'])/len(v['faithfulness'])
                           for v in method_metrics.values()]
        
        ax4.bar([i - width/2 for i in x], compressions_avg, width, 
                label='Compression', alpha=0.7)
        ax4.bar([i + width/2 for i in x], faithfulness_avg, width,
                label='Faithfulness', alpha=0.7)
        ax4.set_title('Metrics by Summarization Method')
        ax4.set_xticks(x)
        ax4.set_xticklabels(method_metrics.keys())
        ax4.legend()
        
        plt.tight_layout()
        
        # Save figure
        fig_file = self.output_dir / "model_comparison.png"
        plt.savefig(fig_file, dpi=300, bbox_inches='tight')
        print(f"✓ Visualization saved to {fig_file}")
        
        plt.close()
    
    def print_comparison_summary(
        self,
        results: List[Dict],
        models: List[str],
        methods: List[str]
    ):
        """Print comparison summary."""
        print("\n" + "="*60)
        print("COMPARISON SUMMARY")
        print("="*60)
        
        for model in models:
            model_short = model.split('/')[-1]
            model_results = [r for r in results if r['model'] == model]
            
            if not model_results:
                continue
            
            print(f"\n{model_short}:")
            
            for method in methods:
                method_results = [r for r in model_results if r['method'] == method]
                
                if not method_results:
                    continue
                
                avg_compression = sum(r['compression_ratio'] for r in method_results) / len(method_results)
                avg_faithfulness = sum(r['faithfulness_score'] for r in method_results) / len(method_results)
                avg_time = sum(r['processing_time'] for r in method_results) / len(method_results)
                
                print(f"  {method:12} | Compression: {avg_compression:.1%} | "
                      f"Faithfulness: {avg_faithfulness:.1%} | Time: {avg_time:.1f}s")
        
        # Overall best
        print("\n" + "-"*60)
        
        best_compression = min(results, key=lambda r: r['compression_ratio'])
        best_faithfulness = max(results, key=lambda r: r['faithfulness_score'])
        fastest = min(results, key=lambda r: r['processing_time'])
        
        print("\n🏆 Best Results:")
        print(f"  Most concise: {best_compression['model'].split('/')[-1]} "
              f"({best_compression['method']}) - {best_compression['compression_ratio']:.1%}")
        print(f"  Most faithful: {best_faithfulness['model'].split('/')[-1]} "
              f"({best_faithfulness['method']}) - {best_faithfulness['faithfulness_score']:.1%}")
        print(f"  Fastest: {fastest['model'].split('/')[-1]} "
              f"({fastest['method']}) - {fastest['processing_time']:.1f}s")


def main():
    """Main execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare summarization models")
    parser.add_argument(
        '--num-docs',
        type=int,
        default=3,
        help='Number of documents to test (default: 3)'
    )
    parser.add_argument(
        '--models',
        nargs='+',
        default=None,
        help='Models to compare'
    )
    
    args = parser.parse_args()
    
    comparison = ModelComparison()
    comparison.compare_models(
        models=args.models,
        num_docs=args.num_docs
    )
    
    print("\n✅ Model comparison complete!")


if __name__ == "__main__":
    main()
