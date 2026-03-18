"""Shared backend helpers for the Streamlit dashboard."""

from __future__ import annotations

import io
import json
import os
from contextlib import redirect_stdout
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from src.embedding.trainer import Trainer
from src.generation.extractive import ExtractiveQA
from src.generation.llm_generator import LLMGenerator
from src.ingestion.chunker import ChunkEngine
from src.ingestion.pdf_extractor import PDFExtractor
from src.preprocessing.nltk_processor import NLTKProcessor
from src.retrieval.retriever import Retriever
from src.retrieval.vector_store import VectorStore


@dataclass
class ActionResult:
    """Result container for ingest/train actions."""

    ok: bool
    message: str
    logs: str
    stats: Optional[Dict] = None


def save_uploaded_pdfs(uploaded_files, pdf_directory: str) -> List[str]:
    """Persist uploaded PDFs to the configured data directory."""
    os.makedirs(pdf_directory, exist_ok=True)
    saved_paths = []
    for uploaded_file in uploaded_files:
        destination = os.path.join(pdf_directory, uploaded_file.name)
        with open(destination, "wb") as f:
            f.write(uploaded_file.getbuffer())
        saved_paths.append(destination)
    return saved_paths


def run_ingest(config: dict, progress_callback: Optional[Callable[[str], None]] = None) -> ActionResult:
    """Execute the ingest pipeline and return logs plus summary stats."""
    logs = io.StringIO()
    stats: Dict[str, int] = {}

    def notify(message: str):
        if progress_callback:
            progress_callback(message)

    try:
        with redirect_stdout(logs):
            extractor = PDFExtractor()
            processor = NLTKProcessor(config)
            chunker = ChunkEngine(
                chunk_size=config["chunk_size"],
                chunk_overlap=config["chunk_overlap"],
                strategy=config.get("chunk_strategy", "sentence"),
            )

            notify("Scanning PDF directory")
            print(f"[INGEST] Scanning: {config['pdf_directory']}")
            documents = extractor.extract_all(config["pdf_directory"])
            notify("Extracting PDF text")
            print(f"[INGEST] Extracted {len(documents)} documents")

            processed_docs = []
            notify("Preprocessing extracted text")
            for doc in documents:
                processed_docs.append({**doc, "processed_text": processor.process(doc["text"])})
            print(f"[INGEST] Preprocessed {len(processed_docs)} documents")

            chunks = []
            notify("Creating chunks")
            for doc in processed_docs:
                raw_chunks = chunker.chunk(doc)
                for chunk in raw_chunks:
                    chunk["processed_text"] = processor.process(chunk["text"])
                chunks.extend(raw_chunks)
            print(f"[INGEST] Created {len(chunks)} chunks")

            processed_chunks_path = config.get(
                "processed_chunks_path", os.path.join("data", "processed", "chunks.json")
            )
            notify("Saving processed chunks")
            os.makedirs(os.path.dirname(processed_chunks_path), exist_ok=True)
            with open(processed_chunks_path, "w", encoding="utf-8") as f:
                json.dump(chunks, f, indent=2)
            print(f"[INGEST] Saved processed chunks to {processed_chunks_path}")

            from src.embedding.embedder import Embedder

            embedder = Embedder(config)
            notify("Generating embeddings")
            vectors = embedder.embed_chunks(chunks)
            print(f"[INGEST] Generated {len(vectors)} embedding vectors")

            notify("Building vector store")
            store = VectorStore(config["vector_store_path"])
            store.add_all(chunks, vectors)
            store.save()
            print(f"[INGEST] Saved to {config['vector_store_path']}")
            print("[INGEST] Done!")
            notify("Ingestion complete")

        unique_sources = sorted({chunk["source"] for chunk in chunks})
        stats = {
            "documents": len(documents),
            "chunks": len(chunks),
            "sources": len(unique_sources),
        }
        return ActionResult(True, "Ingestion completed successfully.", logs.getvalue(), stats)
    except Exception as exc:
        return ActionResult(False, f"Ingestion failed: {exc}", logs.getvalue())


def run_train(config: dict, progress_callback: Optional[Callable[[str], None]] = None) -> ActionResult:
    """Execute the training pipeline and capture console logs."""
    logs = io.StringIO()
    try:
        with redirect_stdout(logs):
            trainer = Trainer(config)
            trainer.train(progress_callback=progress_callback)
            print("[TRAIN] Model saved to:", config["model_save_path"])
        return ActionResult(True, "Training completed successfully.", logs.getvalue())
    except Exception as exc:
        return ActionResult(False, f"Training failed: {exc}", logs.getvalue())


def answer_query(
    config: dict,
    query: str,
    source_filter: Optional[str] = None,
    use_general_llm: bool = False,
    fallback_to_general: bool = True,
) -> Dict:
    """Answer a query through RAG or plain LLM chat depending on the requested mode."""
    generator = LLMGenerator(config)

    if use_general_llm:
        response = generator.chat(query)
        return {
            "mode": "general_llm" if response.get("used_llm") else "failed_general_llm",
            "answer": response["text"],
            "sources": response.get("sources", []),
            "results": [],
        }

    retriever = Retriever(config)
    results = retriever.retrieve(query, source_filter=source_filter)

    if not results:
        if fallback_to_general:
            response = generator.chat(
                query,
                system_prompt=(
                    "You are a helpful AI assistant. "
                    "No relevant document evidence was found, so answer as a general assistant. "
                    "Be honest that the response is not grounded in uploaded PDFs."
                ),
            )
            return {
                "mode": "general_llm_fallback" if response.get("used_llm") else "not_found",
                "answer": response["text"],
                "sources": [],
                "results": [],
            }

        return {
            "mode": "not_found",
            "answer": "Answer not found - no relevant passages above confidence threshold.",
            "sources": [],
            "results": [],
        }

    if config.get("mode", "extractive") == "hybrid":
        response = generator.generate(query, results)
        if response.get("used_llm"):
            return {
                "mode": "rag_llm",
                "answer": response["text"],
                "sources": response.get("sources", []),
                "results": results,
            }

    extractive = ExtractiveQA().answer(query, results)
    return {
        "mode": "extractive",
        "answer": extractive["answer"],
        "sources": extractive.get("sources", []),
        "results": results,
    }


def get_dashboard_status(config: dict) -> Dict:
    """Collect top-level dashboard stats from the project workspace."""
    pdf_directory = config["pdf_directory"]
    processed_chunks_path = config.get("processed_chunks_path", os.path.join("data", "processed", "chunks.json"))
    model_dir = config.get("model_save_path", "models/siamese_bilstm")
    vector_dir = config.get("vector_store_path", "data/vectors")

    pdf_files = []
    if os.path.isdir(pdf_directory):
        pdf_files = sorted(file for file in os.listdir(pdf_directory) if file.lower().endswith(".pdf"))

    chunk_count = 0
    if os.path.exists(processed_chunks_path):
        try:
            with open(processed_chunks_path, "r", encoding="utf-8") as f:
                chunk_count = len(json.load(f))
        except Exception:
            chunk_count = 0

    return {
        "pdf_count": len(pdf_files),
        "pdf_files": pdf_files,
        "chunk_count": chunk_count,
        "vector_store_ready": os.path.exists(os.path.join(vector_dir, "vectors.npz")),
        "trained_model_ready": os.path.exists(os.path.join(model_dir, "encoder.keras")),
    }
