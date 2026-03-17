# AI Research Agent for Private Document Question Answering

An **LLM-powered RAG tool** for querying private PDF document sets, built with **NLTK**, **Keras**, **hybrid retrieval**, and **grounded answer generation**.

This project ingests PDF documents, preprocesses and chunks their text, learns semantic embeddings with a custom Siamese encoder, retrieves relevant passages for a query, and uses an LLM to generate answers grounded in the source material. It supports both API-based models and lightweight local models through Ollama, and falls back to extractive answering if generation is unavailable.

## Table of Contents

- [Quick Start](#quick-start)
- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Configuration](#configuration)
- [Notebook Workflow](#notebook-workflow)
- [Evaluation](#evaluation)
- [Current Capabilities](#current-capabilities)
- [Limitations](#limitations)
- [For Presentation or Viva](#for-presentation-or-viva)

## Quick Start

The fastest local setup uses **Ollama + `phi3:mini`**.

```powershell
py -3.11 -m pip install -r requirements.txt
ollama pull phi3:mini
py -3.11 -m main ingest
py -3.11 -m main train
py -3.11 -m main query "What is Amazon API Gateway?"
```

After the first `ingest` and `train`, you usually only need:

```powershell
py -3.11 -m main query "Your question here"
```

Run `ingest` again only when your PDFs change. Run `train` again when you want to rebuild the embedding model for the updated document set.

## Overview

This repository implements an **LLM-powered Retrieval-Augmented Generation (RAG) pipeline** for private documents, with a strong focus on local ingestion, custom retrieval, and grounded answers.

The system:

- reads PDFs from a local directory
- extracts and preprocesses text with NLTK
- creates chunk-level document representations
- learns custom semantic embeddings using a Siamese Keras model
- combines dense retrieval with lexical retrieval
- uses an LLM to answer from retrieved context
- supports lightweight local generation through Ollama
- falls back to extractive QA if generation is unavailable

In short:

> **PDFs in, evidence retrieved, grounded answers generated.**

## Key Features

- **Private PDF ingestion** using PyMuPDF
- **Classical NLP preprocessing** using NLTK
- **Chunk-based indexing** with raw and processed text separation
- **Custom trainable embedding model** built in Keras
- **Hybrid retrieval** using semantic vectors plus TF-IDF lexical matching
- **LLM-powered answer generation** over retrieved document context
- **Lightweight local LLM support** through Ollama
- **Automatic extractive fallback** when generation is unavailable
- **Custom model experimentation notebook** in [notebooks/custom_model.ipynb](/D:/SIC/notebooks/custom_model.ipynb)
- **Evaluation pipeline** for QA experiments

## System Architecture

```text
PDF Documents
    ->
PDF Extraction
    ->
NLTK Preprocessing
    ->
Chunking
    ->
Embedding + Vector Store
    ->
Hybrid Retrieval
    ->
LLM Answer Generation
```

### Pipeline Summary

```text
User Question
    ->
Query Preprocessing
    ->
Dense + Lexical Retrieval
    ->
Top Relevant Chunks
    ->
Ollama / API LLM
    ->
Grounded Final Answer
```

## How It Works

### 1. Ingestion

When you run `ingest`, the system:

1. scans [data/pdfs](/D:/SIC/data/pdfs) for PDF files
2. extracts text page by page
3. preprocesses the text using NLTK
4. splits documents into smaller chunks
5. stores chunk metadata in `data/processed/chunks.json`
6. embeds the chunks and saves the vector store to `data/vectors`

Core ingestion components:

- [main.py](/D:/SIC/main.py)
- [pdf_extractor.py](/D:/SIC/src/ingestion/pdf_extractor.py)
- [chunker.py](/D:/SIC/src/ingestion/chunker.py)
- [nltk_processor.py](/D:/SIC/src/preprocessing/nltk_processor.py)

### 2. Preprocessing

Each chunk stores:

- `text`: raw readable text for answer display
- `processed_text`: normalized text for training and retrieval

This keeps the learning pipeline clean while preserving readable final answers.

### 3. Training

When you run `train`, the system:

1. loads chunk data from `data/processed/chunks.json`
2. builds a vocabulary
3. creates positive and negative training pairs
4. trains a Siamese text encoder in Keras
5. saves the trained encoder to `models/siamese_bilstm/encoder.keras`
6. rebuilds the vector store using the trained model

Core training files:

- [trainer.py](/D:/SIC/src/embedding/trainer.py)
- [model.py](/D:/SIC/src/embedding/model.py)
- [embedder.py](/D:/SIC/src/embedding/embedder.py)

### 4. Retrieval

When you run `query`, the system:

1. preprocesses the query
2. embeds it with the same encoder used for chunks
3. performs **dense retrieval** using vector similarity
4. performs **lexical retrieval** using TF-IDF
5. merges and reranks the candidate results
6. passes the best passages into the answer generation layer

Retrieval files:

- [vector_store.py](/D:/SIC/src/retrieval/vector_store.py)
- [retriever.py](/D:/SIC/src/retrieval/retriever.py)

### 5. Answer Generation

By default, the project runs in **hybrid RAG mode**:

- retrieval stays local and document-grounded
- the top retrieved chunks are compressed to fit the model context window
- the LLM is prompted to answer only from the retrieved evidence
- if generation is unavailable, the system falls back to extractive QA

Generation files:

- [llm_generator.py](/D:/SIC/src/generation/llm_generator.py)
- [context_optimizer.py](/D:/SIC/src/generation/context_optimizer.py)
- [extractive.py](/D:/SIC/src/generation/extractive.py)

## Project Structure

```text
.
|-- config.yaml
|-- main.py
|-- data/
|   |-- pdfs/
|   |-- processed/
|   `-- vectors/
|-- models/
|   `-- siamese_bilstm/
|-- notebooks/
|   `-- custom_model.ipynb
|-- src/
|   |-- embedding/
|   |-- evaluation/
|   |-- generation/
|   |-- ingestion/
|   |-- preprocessing/
|   `-- retrieval/
`-- tests/
```

### Important Files

- [main.py](/D:/SIC/main.py): CLI entry point
- [config.yaml](/D:/SIC/config.yaml): central configuration
- [custom_model.ipynb](/D:/SIC/notebooks/custom_model.ipynb): notebook for custom model experimentation
- [trainer.py](/D:/SIC/src/embedding/trainer.py): model training pipeline
- [retriever.py](/D:/SIC/src/retrieval/retriever.py): query-time retrieval and reranking
- [llm_generator.py](/D:/SIC/src/generation/llm_generator.py): LLM-powered answer generation
- [extractive.py](/D:/SIC/src/generation/extractive.py): extractive fallback answer selection

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core application logic |
| NLTK | Tokenization, stopword removal, lemmatization, POS tagging |
| TensorFlow / Keras | Siamese embedding model |
| NumPy | Vector operations |
| scikit-learn | TF-IDF and retrieval helpers |
| PyMuPDF | PDF text extraction |
| PyYAML | Configuration loading |
| matplotlib | Evaluation plots |
| Jupyter | Notebook-based model experimentation |
| Ollama / OpenAI / Google | LLM answer generation |

## Getting Started

### 1. Install Dependencies

Use the same Python interpreter for installation and execution.

```bash
py -3.11 -m pip install -r requirements.txt
```

### 2. Add PDFs

Place your PDF files in [data/pdfs](/D:/SIC/data/pdfs).

### 3. Ingest the Documents

```bash
py -3.11 -m main ingest
```

### 4. Train the Embedding Model

```bash
py -3.11 -m main train
```

### 5. Configure an LLM

Recommended local setup with Ollama:

```powershell
ollama pull phi3:mini
```

Make sure the Ollama app or server is running on:

```text
http://localhost:11434
```

The default config is already set to use Ollama:

```yaml
mode: "hybrid"
llm_provider: "ollama"
ollama_model: "phi3:mini"
```

If you want an API model instead, you can use OpenAI or Google.

OpenAI example:

```powershell
$env:OPENAI_API_KEY="your_api_key"
```

Google example:

```powershell
$env:GOOGLE_API_KEY="your_api_key"
```

### 6. Ask Questions

```bash
py -3.11 -m main query "What is machine learning?"
```

If Ollama is not running, or no API key is available for cloud providers, the system still answers using extractive fallback mode.

### 7. Day-to-Day Workflow

For normal use:

1. put PDFs in [data/pdfs](/D:/SIC/data/pdfs)
2. run `py -3.11 -m main ingest`
3. run `py -3.11 -m main train`
4. ask questions with `py -3.11 -m main query "your question"`

After that:

- if only the question changes, just run `query`
- if the PDF set changes, run `ingest` again
- if you want the custom retriever rebuilt for the new data, run `train` after `ingest`

## Usage

### Ingest

```bash
py -3.11 -m main ingest
```

Optional directory override:

```bash
py -3.11 -m main ingest --dir data/pdfs
```

### Train

```bash
py -3.11 -m main train
```

### Query

```bash
py -3.11 -m main query "What is Amazon API Gateway?"
```

Example local-Ollama query:

```bash
py -3.11 -m main query "How is RAG system quality measured?"
```

Use top-k override:

```bash
py -3.11 -m main query "What is Amazon API Gateway?" --top-k 5
```

Filter to a specific source:

```bash
py -3.11 -m main query "How is RAG system quality measured?" --source rag_quality_test.pdf
```

### Evaluate

```bash
py -3.11 -m main evaluate --eval-set evaluation_data/qa_pairs.json
```

### Run Tests

```bash
py -3.11 -m unittest discover -s tests -v
```

## Configuration

All project settings are managed in [config.yaml](/D:/SIC/config.yaml).

### Core Configuration Areas

- **Ingestion**
  - `pdf_directory`

- **Chunking**
  - `chunk_size`
  - `chunk_overlap`
  - `chunk_strategy`

- **Preprocessing**
  - `remove_stopwords`
  - `apply_lemmatization`
  - `apply_pos_tagging`

- **Embedding / Training**
  - `encoder_architecture`
  - `embedding_dim`
  - `lstm_units`
  - `dense_units`
  - `max_sequence_length`
  - `vocab_size`
  - `batch_size`
  - `epochs`

- **Retrieval**
  - `top_k`
  - `similarity_threshold`
  - `rerank_multiplier`
  - `dense_score_weight`
  - `lexical_score_weight`

- **Generation**
  - `mode`
  - `llm_provider`
  - `llm_api_key`
  - `openai_model`
  - `google_model`
  - `ollama_model`
  - `ollama_base_url`
  - `max_context_tokens`

### Recommended RAG Settings

The default config now treats this project as an LLM-powered RAG application:

```yaml
mode: "hybrid"
llm_provider: "ollama"
ollama_model: "phi3:mini"
```

Supported generation modes:

- `hybrid`: retrieve locally, then generate with an LLM
- `extractive`: retrieve locally, then return extracted snippets only

Supported providers:

- `ollama`
- `openai`
- `google`

### Encoder Architectures

You can switch the encoder model in config without changing the training pipeline:

```yaml
encoder_architecture: "bilstm"
```

Supported values:

- `bilstm`
- `bigru`
- `cnn`

## Notebook Workflow

If you want to experiment with your own model design, use [notebooks/custom_model.ipynb](/D:/SIC/notebooks/custom_model.ipynb).

Recommended workflow:

1. run `py -3.11 -m main ingest`
2. open the notebook
3. build or modify the encoder inside the notebook
4. train and save the model artifacts
5. query the system normally from the CLI

This notebook is useful for:

- trying different encoder architectures
- understanding how the embedding model works
- testing custom training ideas without changing the full codebase immediately

## Evaluation

The evaluation pipeline lets you test retrieval and QA behavior on a prepared question-answer set.

Run:

```bash
py -3.11 -m main evaluate --eval-set evaluation_data/qa_pairs.json
```

Outputs are written to the directory defined by `eval_output_dir` in [config.yaml](/D:/SIC/config.yaml).

Relevant files:

- [metrics.py](/D:/SIC/src/evaluation/metrics.py)
- [visualizer.py](/D:/SIC/src/evaluation/visualizer.py)

## Current Capabilities

This system is strongest at:

- searching across multiple PDF documents
- retrieving relevant passages from technical documents
- generating grounded answers from retrieved context
- learning custom semantic embeddings from the ingested corpus
- supporting local experimentation with retrieval and embedding settings
- running with a lightweight local LLM through Ollama

## Limitations

This project is retrieval-centric, so it still has a few important limitations:

- LLM answer quality depends heavily on retrieval quality
- answer quality depends on chunk quality and reranking quality
- very large technical PDFs can still produce imperfect ranking
- the Siamese model learns similarity, not full reasoning
- cloud providers require an external API key
- local Ollama answers depend on the quality and size of the selected local model

## For Presentation or Viva

### One-Line Description

> This is an LLM-powered RAG system for private PDF question answering that combines local document retrieval, a custom Keras embedding model, and grounded answer generation.

### Short Explanation

> The system reads PDFs, preprocesses and chunks their text, learns semantic embeddings for those chunks, retrieves the most relevant passages for a question, and then uses an LLM to generate an answer strictly from the retrieved context.

### Very Short Viva Version

> I built a private-document RAG system. It indexes PDF content locally, retrieves the most relevant chunks using a custom embedding model plus lexical search, and then uses a lightweight LLM to generate an answer from that retrieved evidence.

### Academic Framing

You can describe it like this:

> This project combines classical NLP preprocessing with a neural Siamese embedding model for semantic retrieval over private documents. It uses hybrid retrieval by combining dense vector similarity and lexical matching, then performs LLM-based answer generation over the retrieved context.

### Why This Design Matters

- It is more grounded than plain free-form generation.
- It is more explainable because answers come from retrieved source documents.
- It supports private document collections.
- It allows experimentation with retrieval and embedding models.

---

If you are presenting this project, the best summary is:

**"I built an LLM-powered RAG system that retrieves evidence from private PDF documents and generates grounded answers from that retrieved context."**
