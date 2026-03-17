"""
Retriever
Orchestrates: query rewriting -> preprocessing -> dense search -> lexical search -> hybrid reranking.
"""

import re
from typing import Dict, List

from src.embedding.embedder import Embedder
from src.preprocessing.nltk_processor import NLTKProcessor
from src.retrieval.metadata_filter import MetadataFilter
from src.retrieval.query_rewriter import QueryRewriter
from src.retrieval.vector_store import VectorStore


class Retriever:
    """End-to-end retrieval pipeline with citations."""

    def __init__(self, config: dict):
        self.config = config
        self.processor = NLTKProcessor(config)
        self.embedder = Embedder(config)
        self.store = VectorStore(config["vector_store_path"])
        self.store.load()
        self.rewriter = QueryRewriter(config)
        self.meta_filter = MetadataFilter()

    def retrieve(self, query: str, source_filter: str = None, page_range: tuple = None) -> List[Dict]:
        """Retrieve relevant chunks for a natural language query."""
        expanded_query = self.rewriter.rewrite(query)
        processed = self.processor.process(expanded_query)
        query_vector = self.embedder.embed_text(processed)

        top_k = self.config.get("top_k", 5)
        threshold = self.config.get("similarity_threshold", 0.3)
        search_depth = max(top_k * self.config.get("rerank_multiplier", 4), top_k)

        dense_results = self.store.search(query_vector, top_k=search_depth, threshold=threshold)
        lexical_results = self.store.search_lexical(processed, top_k=search_depth)
        results = self._merge_results(dense_results, lexical_results)

        if source_filter or page_range:
            results = self.meta_filter.filter(results, source=source_filter, page_range=page_range)

        reranked = self._rerank(query, results)
        return reranked[:top_k]

    def _merge_results(self, dense_results: List[Dict], lexical_results: List[Dict]) -> List[Dict]:
        """Merge dense and lexical results keyed by chunk identity."""
        merged: Dict[tuple, Dict] = {}

        for result in dense_results:
            key = (result["source"], result["page"], result["chunk_id"])
            merged[key] = {**result, "dense_score": result["score"], "lexical_score": 0.0}

        for result in lexical_results:
            key = (result["source"], result["page"], result["chunk_id"])
            if key in merged:
                merged[key]["lexical_score"] = max(merged[key]["lexical_score"], result["score"])
            else:
                merged[key] = {**result, "dense_score": 0.0, "lexical_score": result["score"]}

        return list(merged.values())

    def _rerank(self, query: str, results: List[Dict]) -> List[Dict]:
        """Blend dense score, lexical score, and query overlap heuristics."""
        if not results:
            return []

        query_terms = set(re.findall(r"\w+", query.lower()))
        rerank_weight = self.config.get("rerank_overlap_weight", 0.08)
        dense_weight = self.config.get("dense_score_weight", 0.55)
        lexical_weight = self.config.get("lexical_score_weight", 0.45)
        heading_penalty = self.config.get("heading_penalty", 0.2)
        short_penalty = self.config.get("short_chunk_penalty", 0.08)

        max_dense = max((result.get("dense_score", 0.0) for result in results), default=1.0) or 1.0
        max_lexical = max((result.get("lexical_score", 0.0) for result in results), default=1.0) or 1.0

        reranked = []
        for result in results:
            text = result.get("text", "")
            terms = set(re.findall(r"\w+", text.lower()))
            overlap = len(query_terms & terms)

            dense_component = result.get("dense_score", 0.0) / max_dense
            lexical_component = result.get("lexical_score", 0.0) / max_lexical
            adjusted_score = dense_component * dense_weight + lexical_component * lexical_weight
            adjusted_score += overlap * rerank_weight

            lines = [line.strip() for line in text.splitlines() if line.strip()]
            looks_like_heading = len(lines) <= 2 and len(text.split()) < 18
            too_short = len(text.split()) < 35
            if looks_like_heading:
                adjusted_score -= heading_penalty
            elif too_short:
                adjusted_score -= short_penalty

            reranked.append(
                {
                    **result,
                    "score": adjusted_score,
                    "query_overlap": overlap,
                }
            )

        reranked.sort(key=lambda item: item["score"], reverse=True)
        return reranked
