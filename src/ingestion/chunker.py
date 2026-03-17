"""
Chunking Engine
Splits documents into configurable-size chunks with metadata.
Supports fixed-size and sentence-boundary-aware strategies.
"""

from typing import List, Dict
from nltk.tokenize import sent_tokenize


class ChunkEngine:
    """
    Split document text into chunks for embedding.

    Strategies:
        - "fixed": Fixed word count with overlap
        - "sentence": Sentence-boundary-aware grouping
    """

    def __init__(self, chunk_size: int = 300, chunk_overlap: int = 50,
                 strategy: str = "sentence"):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = strategy

    def chunk(self, document: Dict) -> List[Dict]:
        """
        Split a document into chunks with metadata.

        Args:
            document: Dict with "text", "source", "page_num", "processed_text"

        Returns:
            List of chunk dicts with metadata.
        """
        text = document["text"]

        if self.strategy == "sentence":
            chunks = self._sentence_chunk(text)
        else:
            chunks = self._fixed_chunk(text)

        # Attach metadata to each chunk
        result = []
        for i, chunk_text in enumerate(chunks):
            result.append({
                "text": chunk_text,
                "source": document["source"],
                "page": document.get("page_num", 0),
                "chunk_id": i,
            })

        return result

    def _fixed_chunk(self, text: str) -> List[str]:
        """Fixed-size chunking with word-level overlap."""
        words = text.split()
        chunks = []
        start = 0

        while start < len(words):
            end = start + self.chunk_size
            chunk = " ".join(words[start:end])
            if chunk.strip():
                chunks.append(chunk)
            start = end - self.chunk_overlap

        return chunks

    def _sentence_chunk(self, text: str) -> List[str]:
        """Sentence-boundary-aware chunking using NLTK sent_tokenize."""
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence.split())
            if current_length + sentence_length > self.chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                # Keep overlap
                overlap_words = " ".join(current_chunk).split()[-self.chunk_overlap:]
                current_chunk = [" ".join(overlap_words)]
                current_length = len(overlap_words)
            current_chunk.append(sentence)
            current_length += sentence_length

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks
