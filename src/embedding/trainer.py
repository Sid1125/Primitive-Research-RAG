"""
Training Loop for Siamese BiLSTM
Generates positive/negative pairs and trains the embedding model.
"""

import json
import os
from typing import Dict, List

import numpy as np

from src.embedding.vocabulary import Vocabulary
from src.retrieval.vector_store import VectorStore


class Trainer:
    """Train the Siamese BiLSTM embedding model."""

    def __init__(self, config: dict):
        self.config = config
        self.processed_chunks_path = config.get(
            "processed_chunks_path", os.path.join("data", "processed", "chunks.json")
        )

    @staticmethod
    def _training_text(chunk: Dict) -> str:
        """Return the normalized text used for training and embedding."""
        return chunk.get("processed_text", chunk["text"])

    def _expand_chunks_for_training(self, chunks: List[Dict]) -> List[Dict]:
        """Create smaller pseudo-chunks when the corpus is too small to train directly."""
        if len(chunks) > 1:
            return chunks

        expanded = []
        window_size = self.config.get("min_training_window_words", 20)
        stride = self.config.get("min_training_window_stride", 10)

        for chunk in chunks:
            words = chunk.get("text", "").split()
            if len(words) < max(window_size, 4):
                continue

            start = 0
            pseudo_id = 0
            while start < len(words):
                end = min(start + window_size, len(words))
                window_words = words[start:end]
                if len(window_words) < 4:
                    break
                expanded.append(
                    {
                        **chunk,
                        "processed_text": " ".join(window_words),
                        "chunk_id": pseudo_id,
                    }
                )
                pseudo_id += 1
                if end == len(words):
                    break
                start += stride

        if expanded:
            print(
                f"[TRAIN] Expanded {len(chunks)} original chunk(s) into "
                f"{len(expanded)} training windows for small-corpus training"
            )
            return expanded

        return chunks

    def generate_pairs(self, chunks: List[Dict]) -> tuple:
        """
        Generate training pairs from chunks.

        Positive pairs: adjacent chunks from the same source document.
        Negative pairs: chunks from different source documents when possible.
        """
        import random

        chunks = self._expand_chunks_for_training(chunks)
        if len(chunks) < 2:
            return [], [], []

        source_to_chunks = {}
        for chunk in chunks:
            source_to_chunks.setdefault(chunk.get("source", "unknown"), []).append(chunk)

        for doc_chunks in source_to_chunks.values():
            doc_chunks.sort(key=lambda item: (item.get("page", 0), item.get("chunk_id", 0)))

        texts_a = []
        texts_b = []
        labels = []

        for doc_chunks in source_to_chunks.values():
            for i in range(len(doc_chunks) - 1):
                texts_a.append(self._training_text(doc_chunks[i]))
                texts_b.append(self._training_text(doc_chunks[i + 1]))
                labels.append(1)

        num_positives = len(labels)
        sources = list(source_to_chunks.keys())

        if len(sources) > 1:
            for _ in range(num_positives):
                src1, src2 = random.sample(sources, 2)
                chunk1 = random.choice(source_to_chunks[src1])
                chunk2 = random.choice(source_to_chunks[src2])
                texts_a.append(self._training_text(chunk1))
                texts_b.append(self._training_text(chunk2))
                labels.append(0)
        elif sources:
            doc_chunks = source_to_chunks[sources[0]]
            candidate_pairs = [
                (self._training_text(doc_chunks[i]), self._training_text(doc_chunks[j]))
                for i in range(len(doc_chunks))
                for j in range(i + 2, len(doc_chunks))
            ]
            random.shuffle(candidate_pairs)
            for text_a, text_b in candidate_pairs[:num_positives]:
                texts_a.append(text_a)
                texts_b.append(text_b)
                labels.append(0)

        combined = list(zip(texts_a, texts_b, labels))
        random.shuffle(combined)

        if not combined:
            return [], [], []

        text_a, text_b, output_labels = zip(*combined)
        return list(text_a), list(text_b), list(output_labels)

    def _load_chunks(self) -> List[Dict]:
        """Load chunk metadata produced during ingestion."""
        if os.path.exists(self.processed_chunks_path):
            with open(self.processed_chunks_path, "r", encoding="utf-8") as f:
                return json.load(f)

        metadata_path = os.path.join(self.config["vector_store_path"], "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as f:
                return json.load(f)

        raise FileNotFoundError(
            f"Missing processed chunks at {self.processed_chunks_path}. Run ingest first."
        )

    def train(self):
        """Run the full training pipeline."""
        from src.embedding.model import build_siamese_model, contrastive_loss

        chunks = self._load_chunks()
        print(f"[TRAIN] Loaded {len(chunks)} chunks")

        print("[TRAIN] Building vocabulary")
        vocab = Vocabulary(max_size=self.config.get("vocab_size", 20000))
        vocab.build([self._training_text(chunk) for chunk in chunks])

        model_save_path = self.config["model_save_path"]
        os.makedirs(model_save_path, exist_ok=True)
        vocab.save(os.path.join(model_save_path, "vocab.json"))

        print("[TRAIN] Generating pairs")
        texts_a, texts_b, labels = self.generate_pairs(chunks)
        print(f"[TRAIN] Generated {len(labels)} pairs ({sum(labels)} positive)")
        if not labels:
            raise ValueError("Not enough chunk pairs were generated for training.")

        max_len = self.config.get("max_sequence_length", 200)
        seqs_a = np.array([vocab.encode(text, max_len) for text in texts_a])
        seqs_b = np.array([vocab.encode(text, max_len) for text in texts_b])
        labels_arr = np.array(labels, dtype=np.float32)

        print("[TRAIN] Building model")
        siamese, encoder = build_siamese_model(
            vocab_size=vocab.size,
            embedding_dim=self.config.get("embedding_dim", 128),
            lstm_units=self.config.get("lstm_units", 64),
            dense_units=self.config.get("dense_units", 128),
            max_seq_length=max_len,
            architecture=self.config.get("encoder_architecture", "bilstm"),
        )
        siamese.compile(optimizer="adam", loss=contrastive_loss, metrics=["accuracy"])

        print("[TRAIN] Starting training loop")
        validation_split = 0.2 if len(labels_arr) >= 5 else 0.0
        siamese.fit(
            [seqs_a, seqs_b],
            labels_arr,
            batch_size=min(self.config.get("batch_size", 32), len(labels_arr)),
            epochs=self.config.get("epochs", 20),
            validation_split=validation_split,
        )

        encoder_path = os.path.join(model_save_path, "encoder.keras")
        encoder.save(encoder_path)
        print(f"[TRAIN] Model saved to {encoder_path}")

        print("[TRAIN] Rebuilding vector store with trained encoder")
        chunk_sequences = np.array(
            [vocab.encode(self._training_text(chunk), max_len) for chunk in chunks]
        )
        vectors = encoder.predict(chunk_sequences, verbose=0)
        store = VectorStore(self.config["vector_store_path"])
        store.add_all(chunks, vectors)
        store.save()
        print(f"[TRAIN] Vector store saved to {self.config['vector_store_path']}")
