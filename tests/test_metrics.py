import importlib
import json
import os
import shutil
import sys
import types
import unittest
from unittest import mock


class MetricsTests(unittest.TestCase):
    def test_evaluate_restores_top_k_and_aggregates_results(self):
        class FakeRetriever:
            def __init__(self, config):
                self.config = config

            def retrieve(self, query):
                if query == "q1":
                    return [
                        {"source": "doc1.pdf", "page": 1, "score": 0.9},
                        {"source": "doc2.pdf", "page": 2, "score": 0.3},
                    ]
                return [
                    {"source": "doc3.pdf", "page": 3, "score": 0.8},
                    {"source": "doc4.pdf", "page": 4, "score": 0.1},
                ]

        fake_module = types.ModuleType("src.retrieval.retriever")
        fake_module.Retriever = FakeRetriever

        temp_root = os.path.join(os.getcwd(), "_tmp_tests")
        os.makedirs(temp_root, exist_ok=True)
        temp_dir = os.path.join(temp_root, "metrics_test")
        shutil.rmtree(temp_dir, ignore_errors=True)
        os.makedirs(temp_dir, exist_ok=True)
        try:
            eval_set = os.path.join(temp_dir, "qa.json")
            with open(eval_set, "w", encoding="utf-8") as f:
                json.dump(
                    [
                        {"question": "q1", "expected_chunks": ["doc1.pdf:1"]},
                        {"question": "q2", "expected_chunks": ["missing.pdf:9"]},
                    ],
                    f,
                )

            with mock.patch.dict(sys.modules, {"src.retrieval.retriever": fake_module}):
                metrics = importlib.import_module("src.evaluation.metrics")
                metrics = importlib.reload(metrics)

                config = {"top_k": 5}
                results = metrics.evaluate(config, eval_set)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        self.assertEqual(config["top_k"], 5)
        self.assertEqual(results["precision@1"], 0.5)
        self.assertEqual(results["recall@1"], 0.5)
        self.assertEqual(results["mrr"], 0.5)
        self.assertEqual(len(results["per_question"]), 2)


if __name__ == "__main__":
    unittest.main()
