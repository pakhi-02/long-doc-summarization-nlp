"""
Evaluation module for assessing summary quality.
Implements ROUGE scores, BERTScore, and custom metrics.
"""

from typing import Dict, List, Optional
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import numpy as np


class SummaryEvaluator:
    """
    Evaluates summary quality using multiple metrics.
    """

    def __init__(self, metrics: Optional[List[str]] = None):
        """
        Initialize the evaluator.

        Args:
            metrics (List[str]): List of metrics to compute
                Options: ['rouge1', 'rouge2', 'rougeL', 'bertscore']
        """
        self.metrics = metrics or ['rouge1', 'rouge2', 'rougeL']
        
        # Initialize ROUGE scorer if needed
        if any('rouge' in m for m in self.metrics):
            rouge_types = [m for m in self.metrics if 'rouge' in m]
            self.rouge_scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=True)

    def compute_rouge(
        self,
        summary: str,
        reference: str
    ) -> Dict[str, float]:
        """
        Compute ROUGE scores between summary and reference.

        Args:
            summary (str): Generated summary
            reference (str): Reference/gold summary

        Returns:
            Dict[str, float]: ROUGE scores (precision, recall, f1)
        """
        scores = self.rouge_scorer.score(reference, summary)
        
        result = {}
        for metric, score in scores.items():
            result[f"{metric}_precision"] = score.precision
            result[f"{metric}_recall"] = score.recall
            result[f"{metric}_f1"] = score.fmeasure
        
        return result

    def compute_bertscore(
        self,
        summaries: List[str],
        references: List[str],
        lang: str = "en"
    ) -> Dict[str, any]:
        """
        Compute BERTScore for semantic similarity.

        Args:
            summaries (List[str]): List of generated summaries
            references (List[str]): List of reference summaries
            lang (str): Language code

        Returns:
            Dict containing precision, recall, and F1 scores
        """
        P, R, F1 = bert_score(
            summaries,
            references,
            lang=lang,
            verbose=False
        )
        
        return {
            "bertscore_precision": P.mean().item(),
            "bertscore_recall": R.mean().item(),
            "bertscore_f1": F1.mean().item(),
            "bertscore_precision_list": P.tolist(),
            "bertscore_recall_list": R.tolist(),
            "bertscore_f1_list": F1.tolist()
        }

    def evaluate(
        self,
        summary: str,
        reference: str,
        include_bertscore: bool = True
    ) -> Dict[str, float]:
        """
        Comprehensive evaluation of a single summary.

        Args:
            summary (str): Generated summary
            reference (str): Reference summary
            include_bertscore (bool): Whether to compute BERTScore

        Returns:
            Dict[str, float]: All computed metrics
        """
        results = {}
        
        # Compute ROUGE scores
        if any('rouge' in m for m in self.metrics):
            rouge_scores = self.compute_rouge(summary, reference)
            results.update(rouge_scores)
        
        # Compute BERTScore
        if include_bertscore and 'bertscore' in self.metrics:
            bert_scores = self.compute_bertscore([summary], [reference])
            results.update({
                "bertscore_precision": bert_scores["bertscore_precision"],
                "bertscore_recall": bert_scores["bertscore_recall"],
                "bertscore_f1": bert_scores["bertscore_f1"]
            })
        
        # Add basic statistics
        results.update({
            "summary_length": len(summary.split()),
            "reference_length": len(reference.split()),
            "compression_ratio": len(reference.split()) / len(summary.split()) if len(summary.split()) > 0 else 0
        })
        
        return results

    def evaluate_batch(
        self,
        summaries: List[str],
        references: List[str]
    ) -> Dict[str, any]:
        """
        Evaluate multiple summaries at once.

        Args:
            summaries (List[str]): List of generated summaries
            references (List[str]): List of reference summaries

        Returns:
            Dict containing aggregated metrics
        """
        if len(summaries) != len(references):
            raise ValueError("Number of summaries must match number of references")
        
        all_rouge_scores = []
        
        # Compute ROUGE for each pair
        for summary, reference in zip(summaries, references):
            rouge = self.compute_rouge(summary, reference)
            all_rouge_scores.append(rouge)
        
        # Aggregate ROUGE scores
        aggregated = {}
        for key in all_rouge_scores[0].keys():
            values = [scores[key] for scores in all_rouge_scores]
            aggregated[f"{key}_mean"] = np.mean(values)
            aggregated[f"{key}_std"] = np.std(values)
        
        # Compute BERTScore for batch
        if 'bertscore' in self.metrics:
            bert_scores = self.compute_bertscore(summaries, references)
            aggregated.update(bert_scores)
        
        return aggregated


def calculate_faithfulness_score(
    original_text: str,
    summary: str,
    sample_size: int = 5
) -> float:
    """
    Estimate faithfulness by checking key facts/entities preservation.
    This is a simplified heuristic - real faithfulness checking requires
    more sophisticated NLI models.

    Args:
        original_text (str): Original document
        summary (str): Generated summary
        sample_size (int): Number of key terms to check (unused)

    Returns:
        float: Faithfulness ratio (0-1)
    """
    # Extract potential key terms (simplified approach)
    summary_words = set(summary.lower().split())
    original_words = set(original_text.lower().split())
    
    # Check overlap
    common_words = summary_words.intersection(original_words)
    
    # Calculate metrics
    faithfulness_ratio = len(common_words) / len(summary_words) if len(summary_words) > 0 else 0
    
    return faithfulness_ratio


def print_evaluation_report(results: Dict[str, float]):
    """
    Pretty print evaluation results.

    Args:
        results (Dict[str, float]): Evaluation metrics
    """
    print("\n" + "="*60)
    print("EVALUATION REPORT")
    print("="*60)
    
    # ROUGE scores
    if any('rouge' in k for k in results.keys()):
        print("\n📊 ROUGE Scores:")
        for key, value in results.items():
            if 'rouge' in key.lower():
                print(f"  {key:.<40} {value:.4f}")
    
    # BERTScore
    if any('bertscore' in k for k in results.keys()):
        print("\n🎯 BERTScore:")
        for key, value in results.items():
            if 'bertscore' in key.lower() and not key.endswith('_list'):
                print(f"  {key:.<40} {value:.4f}")
    
    # Other metrics
    print("\n📏 Document Statistics:")
    for key in ['summary_length', 'reference_length', 'compression_ratio']:
        if key in results:
            print(f"  {key:.<40} {results[key]:.2f}")
    
    print("="*60 + "\n")
