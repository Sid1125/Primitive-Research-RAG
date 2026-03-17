"""
Query Rewriter
Expands queries using NLTK WordNet synonyms for better retrieval.
"""

from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from typing import List


class QueryRewriter:
    """Expand / reformulate queries using NLTK + WordNet."""

    def __init__(self, config: dict):
        self.enabled = config.get("enable_query_rewriting", True)
        self.max_synonyms = config.get("max_synonyms", 3)

    def rewrite(self, query: str) -> str:
        """Expand query with WordNet synonyms for content words."""
        if not self.enabled:
            return query

        tokens = word_tokenize(query.lower())
        tagged = pos_tag(tokens)
        expanded = list(tokens)

        for word, tag in tagged:
            if tag.startswith(("NN", "VB", "JJ")):  # nouns, verbs, adjectives
                synonyms = self._get_synonyms(word)
                expanded.extend(synonyms[:self.max_synonyms])

        return " ".join(expanded)

    def _get_synonyms(self, word: str) -> List[str]:
        """Get unique synonyms from WordNet synsets."""
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                name = lemma.name().replace("_", " ").lower()
                if name != word:
                    synonyms.add(name)
        return list(synonyms)
