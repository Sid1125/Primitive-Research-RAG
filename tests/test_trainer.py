import unittest
import types
import sys

numpy_stub = types.ModuleType("numpy")
numpy_stub.ndarray = object
numpy_stub.load = lambda *args, **kwargs: None
sys.modules.setdefault("numpy", numpy_stub)

from src.embedding.trainer import Trainer


class TrainerTests(unittest.TestCase):
    def test_generate_pairs_can_expand_single_chunk_for_small_corpus_training(self):
        trainer = Trainer(
            {
                "vector_store_path": "unused",
                "min_training_window_words": 6,
                "min_training_window_stride": 3,
            }
        )
        chunks = [
            {
                "text": "one two three four five six seven eight nine ten eleven twelve",
                "source": "doc1.pdf",
                "page": 1,
                "chunk_id": 0,
            }
        ]

        texts_a, texts_b, labels = trainer.generate_pairs(chunks)

        self.assertGreaterEqual(len(labels), 2)
        self.assertIn(1, labels)

    def test_generate_pairs_creates_balanced_positive_and_negative_examples(self):
        trainer = Trainer({"vector_store_path": "unused"})
        chunks = [
            {"text": "doc1 page1 a", "source": "doc1.pdf", "page": 1, "chunk_id": 0},
            {"text": "doc1 page1 b", "source": "doc1.pdf", "page": 1, "chunk_id": 1},
            {"text": "doc2 page1 a", "source": "doc2.pdf", "page": 1, "chunk_id": 0},
            {"text": "doc2 page1 b", "source": "doc2.pdf", "page": 2, "chunk_id": 1},
        ]

        texts_a, texts_b, labels = trainer.generate_pairs(chunks)

        self.assertEqual(len(texts_a), 4)
        self.assertEqual(len(texts_b), 4)
        self.assertEqual(len(labels), 4)
        self.assertEqual(labels.count(1), 2)
        self.assertEqual(labels.count(0), 2)

    def test_generate_pairs_returns_empty_for_insufficient_chunks(self):
        trainer = Trainer({"vector_store_path": "unused"})

        self.assertEqual(
            trainer.generate_pairs([{"text": "only", "source": "doc.pdf"}]),
            ([], [], []),
        )


if __name__ == "__main__":
    unittest.main()
