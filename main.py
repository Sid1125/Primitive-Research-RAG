"""
AI Research Agent for Private Document Question Answering
using NLTK and Keras

CLI Entry Point
"""

import argparse
import json
import os
import sys

import yaml

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def cmd_ingest(args, config):
    """Ingest PDF documents and persist processed chunks."""
    from src.ingestion.chunker import ChunkEngine
    from src.ingestion.pdf_extractor import PDFExtractor
    from src.preprocessing.nltk_processor import NLTKProcessor
    from src.retrieval.vector_store import VectorStore

    print(f"[INGEST] Scanning: {config['pdf_directory']}")

    extractor = PDFExtractor()
    documents = extractor.extract_all(config["pdf_directory"])
    print(f"[INGEST] Extracted {len(documents)} documents")

    processor = NLTKProcessor(config)
    processed_docs = []
    for doc in documents:
        processed_docs.append({**doc, "processed_text": processor.process(doc["text"])})
    print(f"[INGEST] Preprocessed {len(processed_docs)} documents")

    chunker = ChunkEngine(
        chunk_size=config["chunk_size"],
        chunk_overlap=config["chunk_overlap"],
        strategy=config.get("chunk_strategy", "sentence"),
    )
    chunks = []
    for doc in processed_docs:
        raw_chunks = chunker.chunk(doc)
        for chunk in raw_chunks:
            chunk["processed_text"] = processor.process(chunk["text"])
        chunks.extend(raw_chunks)
    print(f"[INGEST] Created {len(chunks)} chunks")

    processed_chunks_path = config.get(
        "processed_chunks_path", os.path.join("data", "processed", "chunks.json")
    )
    os.makedirs(os.path.dirname(processed_chunks_path), exist_ok=True)
    with open(processed_chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)
    print(f"[INGEST] Saved processed chunks to {processed_chunks_path}")

    from src.embedding.embedder import Embedder

    embedder = Embedder(config)
    vectors = embedder.embed_chunks(chunks)
    print(f"[INGEST] Generated {len(vectors)} embedding vectors")

    store = VectorStore(config["vector_store_path"])
    store.add_all(chunks, vectors)
    store.save()
    print(f"[INGEST] Saved to {config['vector_store_path']}")

    print("[INGEST] Done!")


def cmd_query(args, config):
    """Query the document store."""
    from src.generation.extractive import ExtractiveQA
    from src.generation.llm_generator import LLMGenerator
    from src.retrieval.retriever import Retriever

    retriever = Retriever(config)
    results = retriever.retrieve(args.question, source_filter=args.source)

    if not results:
        print("\nAnswer not found - no relevant passages above confidence threshold.\n")
        return

    print(f"\n{'=' * 60}")
    print(f"Query: {args.question}")
    print(f"{'=' * 60}\n")

    mode = config.get("mode", "extractive")

    if mode == "hybrid":
        answer = LLMGenerator(config).generate(args.question, results)
        print(answer["text"])
        if answer["sources"]:
            print("\nSources:")
            for source in answer["sources"]:
                print(f"- {source}")
        return

    answer = ExtractiveQA().answer(args.question, results)
    print(answer["answer"])
    if answer["sources"]:
        print("\nSources:")
        for source in answer["sources"]:
            print(f"- {source}")

    print()
    #for i, result in enumerate(results, 1):
        #print(f"--- Result {i} (score: {result['score']:.4f}) ---")
        #print(f"Source: {result['source']} | Page: {result['page']}")
        #print(f"{result['text'][:500]}")
        #print()


def cmd_train(args, config):
    """Train the Keras embedding model."""
    from src.embedding.trainer import Trainer

    trainer = Trainer(config)
    trainer.train()
    print("[TRAIN] Model saved to:", config["model_save_path"])


def cmd_evaluate(args, config):
    """Run evaluation metrics."""
    from src.evaluation.metrics import evaluate
    from src.evaluation.visualizer import plot_results

    results = evaluate(config, args.eval_set)
    plot_results(results, config["eval_output_dir"])
    print(f"[EVAL] Results saved to {config['eval_output_dir']}")


def main():
    parser = argparse.ArgumentParser(
        description="AI Research Agent - Private Document QA using NLTK & Keras",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config file")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    sub_ingest = subparsers.add_parser("ingest", help="Ingest PDFs into vector store")
    sub_ingest.add_argument("--dir", help="Override PDF directory")

    sub_query = subparsers.add_parser("query", help="Ask a question")
    sub_query.add_argument("question", type=str, help="Your question")
    sub_query.add_argument("--top-k", type=int, default=5, help="Number of results")
    sub_query.add_argument("--source", type=str, help="Filter by source document")

    subparsers.add_parser("train", help="Train the embedding model")

    sub_eval = subparsers.add_parser("evaluate", help="Run evaluation")
    sub_eval.add_argument("--eval-set", required=True, help="Path to QA pairs JSON")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    config = load_config(args.config)

    if args.command == "ingest" and args.dir:
        config["pdf_directory"] = args.dir
    if args.command == "query" and args.top_k:
        config["top_k"] = args.top_k

    commands = {
        "ingest": cmd_ingest,
        "query": cmd_query,
        "train": cmd_train,
        "evaluate": cmd_evaluate,
    }
    commands[args.command](args, config)


if __name__ == "__main__":
    main()
