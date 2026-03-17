"""
Evaluation Visualizer
Charts for precision@k, similarity distributions, and QA results.
"""

import os
from typing import Dict


def plot_results(results: Dict, output_dir: str):
    """
    Generate evaluation charts.

    Produces:
        - precision_at_k.png
        - similarity_distribution.png
        - qa_results.json
    """
    import json
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Bar chart: precision@k and recall@k
    ks = [1, 3, 5, 10]
    precisions = [results.get(f"precision@{k}", 0) for k in ks]
    recalls = [results.get(f"recall@{k}", 0) for k in ks]
    
    x = range(len(ks))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar([i - width/2 for i in x], precisions, width, label='Precision')
    ax.bar([i + width/2 for i in x], recalls, width, label='Recall')
    
    ax.set_ylabel('Scores')
    ax.set_title('Precision and Recall @ k')
    ax.set_xticks(x)
    ax.set_xticklabels([f'k={k}' for k in ks])
    ax.legend()
    
    fig.savefig(os.path.join(output_dir, "precision_at_k.png"))
    plt.close(fig)
    
    # 2. Histogram: cosine similarity distribution
    sims = results.get("similarities", [])
    if sims:
        plt.figure(figsize=(8, 6))
        plt.hist(sims, bins=20, color='skyblue', edgecolor='black')
        plt.title('Cosine Similarity Distribution')
        plt.xlabel('Cosine Similarity')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(output_dir, "similarity_distribution.png"))
        plt.close()
        
    # 3. Save detailed results JSON
    save_data = {k: v for k, v in results.items() if k != "similarities"}
    with open(os.path.join(output_dir, "qa_results.json"), "w") as f:
        json.dump(save_data, f, indent=2)
