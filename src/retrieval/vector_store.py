"""
NumPy-based Vector Store
Stores dense embedding vectors and a lexical TF-IDF index for hybrid retrieval.
"""

import json
import os
import pickle
from typing import Dict, List, Optional

import numpy as np


class VectorStore:
    """
    Vector index backed by NumPy arrays plus a lexical TF-IDF matrix.

    Storage format:
      - vectors.npz for dense vectors
      - metadata.json for chunk metadata
      - lexical_matrix.pkl for TF-IDF features
      - lexical_vectorizer.pkl for TF-IDF vocabulary/config
    """

    def __init__(self, store_path: str = "data/vectors"):
        self.store_path = store_path
        self.vectors: Optional[np.ndarray] = None
        self.metadata: List[Dict] = []
        self.lexical_vectorizer = None
        self.lexical_matrix = None

    def add_all(self, chunks: List[Dict], vectors: np.ndarray):
        """Add chunks with their embedding vectors."""
        self.vectors = vectors
        self.metadata = [
            {
                "text": chunk["text"],
                "processed_text": chunk.get("processed_text"),
                "source": chunk["source"],
                "page": chunk["page"],
                "chunk_id": chunk["chunk_id"],
            }
            for chunk in chunks
        ]
        self._build_lexical_index()

    def search(self, query_vector: np.ndarray, top_k: int = 5, threshold: float = 0.0) -> List[Dict]:
        """Find top-k most similar chunks using dense cosine similarity."""
        if self.vectors is None or len(self.vectors) == 0:
            return []

        similarities = np.dot(self.vectors, query_vector)
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score >= threshold:
                results.append({**self.metadata[idx], "score": score})

        return results

    def search_lexical(self, query_text: str, top_k: int = 5) -> List[Dict]:
        """Find top-k most similar chunks using TF-IDF lexical similarity."""
        if self.lexical_vectorizer is None or self.lexical_matrix is None or not self.metadata:
            return []

        query_matrix = self.lexical_vectorizer.transform([query_text])
        similarities = (self.lexical_matrix @ query_matrix.T).toarray().ravel()
        if similarities.size == 0:
            return []

        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score > 0:
                results.append({**self.metadata[idx], "score": score})
        return results

    def _build_lexical_index(self):
        """Build the lexical TF-IDF index from stored metadata."""
        documents = [item.get("processed_text") or item["text"] for item in self.metadata]
        if not documents:
            self.lexical_vectorizer = None
            self.lexical_matrix = None
            return

        from sklearn.feature_extraction.text import TfidfVectorizer

        self.lexical_vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_features=50000,
        )
        self.lexical_matrix = self.lexical_vectorizer.fit_transform(documents)

    def save(self):
        """Save dense vectors, metadata, and lexical index to disk."""
        os.makedirs(self.store_path, exist_ok=True)
        np.savez(os.path.join(self.store_path, "vectors.npz"), vectors=self.vectors)
        with open(os.path.join(self.store_path, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(self.metadata, f)

        if self.lexical_matrix is not None and self.lexical_vectorizer is not None:
            with open(os.path.join(self.store_path, "lexical_matrix.pkl"), "wb") as f:
                pickle.dump(self.lexical_matrix, f)
            with open(os.path.join(self.store_path, "lexical_vectorizer.pkl"), "wb") as f:
                pickle.dump(self.lexical_vectorizer, f)

    def load(self):
        """Load dense vectors, metadata, and lexical index from disk."""
        vectors_path = os.path.join(self.store_path, "vectors.npz")
        metadata_path = os.path.join(self.store_path, "metadata.json")
        lexical_matrix_path = os.path.join(self.store_path, "lexical_matrix.pkl")
        lexical_vectorizer_path = os.path.join(self.store_path, "lexical_vectorizer.pkl")

        if not os.path.exists(vectors_path) or not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"Vector store not found in {self.store_path}. Run ingest and train first."
            )

        data = np.load(vectors_path)
        self.vectors = data["vectors"]
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        if os.path.exists(lexical_matrix_path) and os.path.exists(lexical_vectorizer_path):
            with open(lexical_matrix_path, "rb") as f:
                self.lexical_matrix = pickle.load(f)
            with open(lexical_vectorizer_path, "rb") as f:
                self.lexical_vectorizer = pickle.load(f)
        else:
            self._build_lexical_index()
