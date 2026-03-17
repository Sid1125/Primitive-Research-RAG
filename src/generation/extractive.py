"""
Extractive QA
Ranks snippets from retrieved chunks and returns concise passages.
Pure NLP - no LLM dependency.
"""

import re
from typing import Dict, List

from nltk.tokenize import sent_tokenize


class ExtractiveQA:
    """Extractive question answering over retrieved chunks."""

    def answer(self, query: str, retrieved_chunks: List[Dict], query_vector=None, embedder=None) -> Dict:
        """Generate an extractive answer from retrieved chunks."""
        if not retrieved_chunks:
            return {
                "answer": "Answer not found - no relevant passages above confidence threshold.",
                "sources": [],
                "confidence": 0.0,
            }

        query_terms = set(re.findall(r"\w+", query.lower()))
        definition_query = self._is_definition_query(query)

        candidates = []
        for chunk in retrieved_chunks:
            for snippet in self._extract_candidates(chunk["text"]):
                cleaned = self._clean_snippet(snippet)
                normalized = self._normalize_text(cleaned)
                if not normalized:
                    continue

                candidates.append(
                    {
                        "text": cleaned,
                        "normalized": normalized,
                        "source": chunk["source"],
                        "page": chunk["page"],
                        "chunk_score": chunk["score"],
                        "query_overlap": self._query_overlap(normalized, query_terms),
                        "definition_bonus": self._definition_bonus(cleaned, query, definition_query),
                        "example_penalty": self._example_penalty(cleaned),
                    }
                )

        candidates.sort(
            key=lambda item: (
                item["definition_bonus"] - item["example_penalty"],
                item["query_overlap"],
                item["chunk_score"],
                -len(item["text"]),
            ),
            reverse=True,
        )

        unique = []
        seen = set()
        for candidate in candidates:
            if candidate["normalized"] in seen:
                continue
            seen.add(candidate["normalized"])
            unique.append(candidate)

        if not unique:
            return {
                "answer": "Answer not found - no relevant passages above confidence threshold.",
                "sources": [],
                "confidence": 0.0,
            }

        top = [unique[0]]
        if len(unique) > 1 and unique[1]["query_overlap"] >= max(2, unique[0]["query_overlap"] - 1):
            if unique[1]["source"] != unique[0]["source"] or unique[1]["page"] != unique[0]["page"]:
                if not self._is_near_duplicate(unique[0]["text"], unique[1]["text"]):
                    top.append(unique[1])

        answer_text = self._finalize_answer(" ".join(candidate["text"] for candidate in top))
        sources = sorted({f"{candidate['source']} (p.{candidate['page']})" for candidate in top})

        return {
            "answer": answer_text,
            "sources": sources,
            "confidence": top[0]["chunk_score"],
        }

    def _extract_candidates(self, text: str) -> List[str]:
        """Break chunk text into smaller deduplicated snippet candidates."""
        candidates = []
        seen = set()

        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            for sentence in sent_tokenize(line):
                cleaned = self._clean_snippet(sentence)
                normalized = self._normalize_text(cleaned)
                if normalized and normalized not in seen:
                    seen.add(normalized)
                    candidates.append(cleaned)

        if not candidates:
            cleaned = self._clean_snippet(text)
            if cleaned:
                candidates.append(cleaned)

        return candidates

    @staticmethod
    def _clean_snippet(text: str) -> str:
        """Collapse whitespace, trim incomplete tails, and avoid oversized snippets."""
        cleaned = re.sub(r"\s+", " ", text).strip()
        cleaned = re.sub(r"\b(and|or|to|of|for|with|in|on|at|by|a|an|the)\s*$", "", cleaned, flags=re.IGNORECASE)
        if len(cleaned) > 260:
            cleaned = cleaned[:260].rsplit(" ", 1)[0]
        return cleaned.strip(" -,:;")

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize text for deduplication."""
        return re.sub(r"\s+", " ", text).strip().lower()

    @staticmethod
    def _query_overlap(text: str, query_terms: set) -> int:
        """Count shared tokens between the query and a candidate snippet."""
        snippet_terms = set(re.findall(r"\w+", text))
        return len(snippet_terms & query_terms)

    @staticmethod
    def _is_definition_query(query: str) -> bool:
        """Detect short definition-style questions."""
        lowered = query.lower().strip()
        return lowered.startswith("what is") or lowered.startswith("what are") or lowered.startswith("define ")

    @staticmethod
    def _definition_bonus(text: str, query: str, definition_query: bool) -> int:
        """Boost snippets that look like definitions when the question is definitional."""
        if not definition_query:
            return 0

        lowered = text.lower()
        query_nouns = re.findall(r"\b[a-zA-Z][\w-]*\b", query.lower())[2:]
        subject_hit = any(term in lowered for term in query_nouns)
        looks_definition_like = any(
            phrase in lowered
            for phrase in [" is a ", " is an ", " refers to ", " is the ", " enables ", " allows "]
        )
        return int(subject_hit) + int(looks_definition_like)

    @staticmethod
    def _example_penalty(text: str) -> int:
        """Penalize snippets that look like code, examples, or transport-level payloads."""
        lowered = text.lower()
        markers = [
            "http/1.1",
            "content-type",
            "host:",
            "accept:",
            "base64",
            "curl ",
            "aws ",
            "{",
            "}",
            "example",
        ]
        return sum(1 for marker in markers if marker in lowered)

    @staticmethod
    def _is_near_duplicate(first: str, second: str) -> bool:
        """Detect when two snippets are effectively the same answer fragment."""
        a = ExtractiveQA._normalize_text(first)
        b = ExtractiveQA._normalize_text(second)
        if a == b:
            return True
        shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
        return shorter in longer and len(shorter) >= max(24, int(len(longer) * 0.6))

    @staticmethod
    def _finalize_answer(text: str) -> str:
        """Trim awkward duplication and incomplete trailing fragments."""
        cleaned = re.sub(r"\s+", " ", text).strip()
        cleaned = re.sub(r"(.{20,}?) \1\b", r"\1", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\b(and|or|to|of|for|with|in|on|at|by|a|an|the)\s*$", "", cleaned, flags=re.IGNORECASE)
        return cleaned.strip(" -,:;")
