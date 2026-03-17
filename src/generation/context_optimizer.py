"""
Context Window Optimizer
Smart truncation and reranking to fit LLM context limits.
"""

from typing import List, Dict


class ContextOptimizer:
    """Optimize retrieved chunks to fit within LLM context window."""

    def __init__(self, max_tokens: int = 4000):
        self.max_tokens = max_tokens

    def optimize(self, chunks: List[Dict]) -> List[Dict]:
        """
        Select and truncate chunks to fit context window.

        Strategy:
            1. Sort by relevance score (highest first)
            2. Accumulate until token budget is exhausted
            3. Deduplicate overlapping content
        """
        # Rough token estimate: 1 token ≈ 0.75 words
        budget = int(self.max_tokens * 0.75)

        optimized = []
        total_words = 0
        seen_text = set()

        for chunk in sorted(chunks, key=lambda c: c["score"], reverse=True):
            text = chunk["text"]
            # Simple dedup
            if text in seen_text:
                continue
            seen_text.add(text)

            words = len(text.split())
            if total_words + words > budget:
                # Truncate last chunk to fit
                remaining = budget - total_words
                if remaining > 20:
                    truncated = " ".join(text.split()[:remaining])
                    optimized.append({**chunk, "text": truncated})
                break

            optimized.append(chunk)
            total_words += words

        return optimized
