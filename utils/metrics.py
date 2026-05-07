# metrics.py
from rouge_score import rouge_scorer

def calculate_rouge_score(machine_text, human_text):
    """
    Computes ROUGE Score between AI-generated and human-written text.
    :param machine_text: AI-generated literature review
    :param human_text: Human-written literature review
    :return: ROUGE-1, ROUGE-2, ROUGE-L scores
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(human_text, machine_text)
    return {
        "ROUGE-1": round(scores['rouge1'].fmeasure, 4),
        "ROUGE-2": round(scores['rouge2'].fmeasure, 4),
        "ROUGE-L": round(scores['rougeL'].fmeasure, 4)
    }
