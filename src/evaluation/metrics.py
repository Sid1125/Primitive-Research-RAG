"""
Evaluation Metrics
Precision@k, Recall@k, MRR, and cosine similarity analysis.
"""

import json
from typing import List, Dict


def evaluate(config: dict, eval_set_path: str) -> Dict:
    """
    Run evaluation using a QA pairs JSON file.

    Expected format:
    [
        {"question": "...", "expected_chunks": ["source:page"], "answer": "..."},
        ...
    ]

    Returns:
        Dict with precision_at_k, recall_at_k, mrr, per_question results.
    """
    from src.retrieval.retriever import Retriever

    with open(eval_set_path, 'r') as f:
        qa_pairs = json.load(f)

    metrics = {
        "precision@1": [], "precision@3": [], "precision@5": [], "precision@10": [],
        "recall@1": [], "recall@3": [], "recall@5": [], "recall@10": [],
        "mrr": [],
        "similarities": [],
        "per_question": []
    }
    
    ks = [1, 3, 5, 10]
    
    # Temporarily override top_k for evaluation depth
    original_top_k = config.get("top_k", 5)
    config["top_k"] = 10
    try:
        retriever = Retriever(config)

        for qa in qa_pairs:
            query = qa["question"]
            expected = qa["expected_chunks"]

            results = retriever.retrieve(query)

            retrieved_ids = [f"{r['source']}:{r['page']}" for r in results]

            sims = [r["score"] for r in results]
            metrics["similarities"].extend(sims)

            q_result = {
                "query": query,
                "retrieved": retrieved_ids,
                "expected": expected,
                "metrics": {}
            }

            for k in ks:
                p_k = precision_at_k(retrieved_ids, expected, k)
                r_k = recall_at_k(retrieved_ids, expected, k)
                metrics[f"precision@{k}"].append(p_k)
                metrics[f"recall@{k}"].append(r_k)
                q_result["metrics"][f"precision@{k}"] = p_k
                q_result["metrics"][f"recall@{k}"] = r_k

            mrr = mean_reciprocal_rank(retrieved_ids, expected)
            metrics["mrr"].append(mrr)
            q_result["metrics"]["mrr"] = mrr

            metrics["per_question"].append(q_result)

        aggregated = {
            "mrr": sum(metrics["mrr"]) / len(metrics["mrr"]) if metrics["mrr"] else 0.0,
            "similarities": metrics["similarities"],
            "per_question": metrics["per_question"]
        }
        for k in ks:
            n = len(qa_pairs)
            aggregated[f"precision@{k}"] = sum(metrics[f"precision@{k}"]) / n if n > 0 else 0.0
            aggregated[f"recall@{k}"] = sum(metrics[f"recall@{k}"]) / n if n > 0 else 0.0

        return aggregated
    finally:
        config["top_k"] = original_top_k


def precision_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """Calculate precision@k."""
    retrieved_k = retrieved[:k]
    hits = len(set(retrieved_k) & set(relevant))
    return hits / k if k > 0 else 0.0


def recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """Calculate recall@k."""
    retrieved_k = retrieved[:k]
    hits = len(set(retrieved_k) & set(relevant))
    return hits / len(relevant) if relevant else 0.0


def mean_reciprocal_rank(retrieved: List[str], relevant: List[str]) -> float:
    """Calculate MRR."""
    for i, item in enumerate(retrieved):
        if item in relevant:
            return 1.0 / (i + 1)
    return 0.0
