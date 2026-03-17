import io
import json
import os
import shutil
import sys
import types
import unittest
from contextlib import redirect_stdout
from unittest import mock

yaml_stub = types.ModuleType("yaml")
yaml_stub.safe_load = lambda stream: {}
sys.modules.setdefault("yaml", yaml_stub)

import main


class MainTests(unittest.TestCase):
    def test_cmd_ingest_saves_chunks_even_without_trained_model(self):
        class FakeExtractor:
            def extract_all(self, directory):
                return [{"text": "Alpha Beta", "source": "doc.pdf", "page_num": 1}]

        class FakeProcessor:
            def __init__(self, config):
                self.config = config

            def process(self, text):
                return text.lower()

        class FakeChunkEngine:
            def __init__(self, chunk_size, chunk_overlap, strategy):
                self.chunk_size = chunk_size
                self.chunk_overlap = chunk_overlap
                self.strategy = strategy

            def chunk(self, doc):
                return [
                    {
                        "text": doc["processed_text"],
                        "source": doc["source"],
                        "page": doc["page_num"],
                        "chunk_id": 0,
                    }
                ]

        class FakeVectorStore:
            def __init__(self, path):
                self.path = path

            def add_all(self, chunks, vectors):
                self.chunks = chunks
                self.vectors = vectors

            def save(self):
                self.saved = True

        class FakeEmbedder:
            def __init__(self, config):
                self.config = config

            def embed_chunks(self, chunks):
                return [[0.1, 0.2] for _ in chunks]

        extractor_module = types.ModuleType("src.ingestion.pdf_extractor")
        extractor_module.PDFExtractor = FakeExtractor
        processor_module = types.ModuleType("src.preprocessing.nltk_processor")
        processor_module.NLTKProcessor = FakeProcessor
        chunker_module = types.ModuleType("src.ingestion.chunker")
        chunker_module.ChunkEngine = FakeChunkEngine
        vector_store_module = types.ModuleType("src.retrieval.vector_store")
        vector_store_module.VectorStore = FakeVectorStore
        embedder_module = types.ModuleType("src.embedding.embedder")
        embedder_module.Embedder = FakeEmbedder

        temp_root = os.path.join(os.getcwd(), "_tmp_tests")
        os.makedirs(temp_root, exist_ok=True)
        temp_dir = os.path.join(temp_root, "main_test")
        shutil.rmtree(temp_dir, ignore_errors=True)
        os.makedirs(temp_dir, exist_ok=True)
        try:
            output_path = os.path.join(temp_dir, "processed", "chunks.json")
            config = {
                "pdf_directory": os.path.join(temp_dir, "pdfs"),
                "chunk_size": 300,
                "chunk_overlap": 50,
                "chunk_strategy": "sentence",
                "processed_chunks_path": output_path,
                "model_save_path": os.path.join(temp_dir, "models"),
                "vector_store_path": os.path.join(temp_dir, "vectors"),
            }
            args = types.SimpleNamespace()
            patched_modules = {
                "src.ingestion.pdf_extractor": extractor_module,
                "src.preprocessing.nltk_processor": processor_module,
                "src.ingestion.chunker": chunker_module,
                "src.retrieval.vector_store": vector_store_module,
                "src.embedding.embedder": embedder_module,
            }

            with mock.patch.dict(sys.modules, patched_modules):
                buffer = io.StringIO()
                with redirect_stdout(buffer):
                    main.cmd_ingest(args, config)

            with open(output_path, "r", encoding="utf-8") as f:
                saved_chunks = json.load(f)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        self.assertEqual(saved_chunks[0]["text"], "alpha beta")
        self.assertIn("Generated 1 embedding vectors", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()
