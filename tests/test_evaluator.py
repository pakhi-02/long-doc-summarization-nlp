from src.evaluator import SummaryEvaluator, calculate_faithfulness_score


def test_compute_rouge_keys_present():
    evaluator = SummaryEvaluator(metrics=["rouge1", "rouge2", "rougeL"])
    generated = "Climate change causes extreme weather and rising sea levels."
    reference = "Rising sea levels and extreme weather are effects of climate change."

    scores = evaluator.compute_rouge(generated, reference)

    assert "rouge1_f1" in scores
    assert "rouge2_f1" in scores
    assert "rougeL_f1" in scores


def test_faithfulness_in_range():
    original = "The quick brown fox jumps over the lazy dog"
    summary = "quick fox jumps"

    score = calculate_faithfulness_score(original_text=original, summary=summary)

    assert 0.0 <= score <= 1.0
