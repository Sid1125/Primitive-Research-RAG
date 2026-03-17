"""
Inference Embedder
Loads trained Keras encoder and converts text to embedding vectors.
Falls back to TF-IDF if no trained model exists.
"""

import os
import pickle
from typing import Dict, List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class Embedder:
    """Convert preprocessed text into embedding vectors using trained encoder or TF-IDF fallback."""

    def __init__(self, config: dict):
        self.config = config
        self.max_seq_length = config.get("max_sequence_length", 200)
        self.model_path = config.get("model_save_path", "models/siamese_bilstm")
        self.embedding_dim = config.get("dense_units", 128)
        self.vectorizer_path = os.path.join(self.model_path, "tfidf_vectorizer.pkl")
        self._encoder = None
        self._vocab = None
        self._vectorizer = None
        self._use_fallback = False

    def _load_model(self):
        """Lazy-load the trained encoder model, or fall back to TF-IDF."""
        if self._encoder is not None or self._use_fallback:
            return

        encoder_path = os.path.join(self.model_path, "encoder.keras")
        vocab_path = os.path.join(self.model_path, "vocab.json")
        if os.path.exists(encoder_path) and os.path.exists(vocab_path):
            try:
                import keras

                from src.embedding.model import L2Normalization
                from src.embedding.vocabulary import Vocabulary

                self._encoder = keras.models.load_model(
                    encoder_path,
                    custom_objects={"L2Normalization": L2Normalization},
                    safe_mode=False,
                    compile=False,
                )
                self._vocab = Vocabulary(self.config.get("vocab_size", 20000))
                self._vocab.load(vocab_path)
                print("[EMBED] Loaded trained Keras encoder")
                return
            except Exception as e:
                print(f"[EMBED] Failed to load Keras model: {e}")

        if os.path.exists(self.vectorizer_path):
            with open(self.vectorizer_path, "rb") as f:
                self._vectorizer = pickle.load(f)
            self._use_fallback = True
            print("[EMBED] Loaded persisted TF-IDF fallback vectorizer")
            return

        self._use_fallback = True

    def embed_text(self, text: str) -> np.ndarray:
        """Convert a single text string to an embedding vector."""
        self._load_model()
        if self._use_fallback:
            if self._vectorizer is None:
                self._vectorizer = TfidfVectorizer(max_features=self.embedding_dim)
                self._vectorizer.fit([text])

            vec = self._vectorizer.transform([text]).toarray()[0]
            if len(vec) < self.embedding_dim:
                vec = np.pad(vec, (0, self.embedding_dim - len(vec)))
            return vec[:self.embedding_dim]

        encoded = self._vocab.encode(text, self.max_seq_length)
        input_array = np.array([encoded])
        vector = self._encoder.predict(input_array, verbose=0)
        return vector[0]

    def embed_chunks(self, chunks: List[Dict]) -> np.ndarray:
        """Embed a list of chunks."""
        self._load_model()
        texts = [chunk.get("processed_text", chunk.get("text", "")) for chunk in chunks]

        if self._use_fallback:
            self._vectorizer = TfidfVectorizer(max_features=self.embedding_dim)
            tfidf_matrix = self._vectorizer.fit_transform(texts).toarray()
            os.makedirs(self.model_path, exist_ok=True)
            with open(self.vectorizer_path, "wb") as f:
                pickle.dump(self._vectorizer, f)

            n_features = tfidf_matrix.shape[1]
            if n_features < self.embedding_dim:
                padding = np.zeros((len(texts), self.embedding_dim - n_features))
                tfidf_matrix = np.hstack([tfidf_matrix, padding])
            return tfidf_matrix[:, :self.embedding_dim]

        encoded = [self._vocab.encode(text, self.max_seq_length) for text in texts]
        input_array = np.array(encoded)
        return self._encoder.predict(input_array, verbose=0)
