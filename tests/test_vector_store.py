import unittest
import os
import shutil
import sys
import types

numpy_stub = types.ModuleType("numpy")
numpy_stub.ndarray = object
numpy_stub.load = lambda *args, **kwargs: None
sys.modules.setdefault("numpy", numpy_stub)

from src.retrieval.vector_store import VectorStore


class VectorStoreTests(unittest.TestCase):
    def test_load_raises_clear_error_when_store_is_missing(self):
        temp_root = os.path.join(os.getcwd(), "_tmp_tests")
        os.makedirs(temp_root, exist_ok=True)
        temp_dir = os.path.join(temp_root, "vector_store_test")
        shutil.rmtree(temp_dir, ignore_errors=True)
        os.makedirs(temp_dir, exist_ok=True)
        try:
            store = VectorStore(temp_dir)
            with self.assertRaisesRegex(FileNotFoundError, "Run ingest and train first"):
                store.load()
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
