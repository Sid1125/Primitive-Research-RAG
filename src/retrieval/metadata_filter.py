"""
Metadata Filter
Filters retrieved chunks by source document, page range, or tags.
"""

from typing import List, Dict, Optional, Tuple


class MetadataFilter:
    """Post-retrieval metadata filtering."""

    def filter(self, results: List[Dict], source: Optional[str] = None,
               page_range: Optional[Tuple[int, int]] = None,
               tags: Optional[List[str]] = None) -> List[Dict]:
        """
        Filter results by metadata criteria.

        Args:
            results: Retrieved chunks with metadata.
            source: Filter by source document filename.
            page_range: (start_page, end_page) inclusive.
            tags: Filter by user-defined tags (future use).

        Returns:
            Filtered list of results.
        """
        filtered = results

        if source:
            filtered = [r for r in filtered
                        if r.get("source", "").lower() == source.lower()]

        if page_range:
            start, end = page_range
            filtered = [r for r in filtered
                        if start <= r.get("page", 0) <= end]

        return filtered
