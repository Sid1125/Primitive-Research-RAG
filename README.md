# AI Research Agent for Private Document Question Answering

A local document question-answering system built with **NLTK**, **Keras**, and **hybrid retrieval**.

This project ingests PDF documents, preprocesses and chunks their text, learns semantic embeddings with a custom Siamese encoder, retrieves relevant passages for a query, and returns grounded answers from the source material.

## Table of Contents

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

## Overview

This repository implements a **RAG-style private document QA pipeline** with a strong focus on local processing and explainable retrieval.

Instead of relying only on a hosted LLM, the system:

- reads PDFs from a local directory
- extracts and preprocesses text with NLTK
- creates chunk-level document representations
- learns custom semantic embeddings using a Siamese Keras model
- combines dense retrieval with lexical retrieval
- returns extractive answers grounded in the retrieved passages

In short:

> **PDFs in, searchable knowledge base out, grounded answers back.**

## Key Features

- **Local PDF ingestion** using PyMuPDF
- **Classical NLP preprocessing** using NLTK
- **Chunk-based document indexing** with raw and processed text separation
- **Custom trainable embedding model** built in Keras
- **Hybrid retrieval** using semantic vectors plus TF-IDF lexical matching
- **Extractive question answering** for source-grounded responses
- **Optional hybrid LLM generation** through config
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
Extractive / Hybrid Answering
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

The ingestion entry point lives in [main.py](/D:/SIC/main.py), and the core ingestion components are:

- [pdf_extractor.py](/D:/SIC/src/ingestion/pdf_extractor.py)
- [chunker.py](/D:/SIC/src/ingestion/chunker.py)
- [nltk_processor.py](/D:/SIC/src/preprocessing/nltk_processor.py)

### 2. Preprocessing

Each chunk keeps two text forms:

- `text`: raw readable text used for final output
- `processed_text`: normalized text used for training and retrieval

This design lets the system learn from cleaner text while still showing readable answers to users.

### 3. Training

When you run `train`, the system:

1. loads chunk data from `data/processed/chunks.json`
2. builds a vocabulary
3. creates positive and negative training pairs
4. trains a Siamese text encoder in Keras
5. saves the trained encoder to `models/siamese_bilstm/encoder.keras`
6. rebuilds the vector store using the trained model

The training pipeline is mainly implemented in:

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
6. sends the top passages to the answering layer

This retrieval flow is implemented in:

- [vector_store.py](/D:/SIC/src/retrieval/vector_store.py)
- [retriever.py](/D:/SIC/src/retrieval/retriever.py)

### 5. Answer Generation

By default, the project uses **extractive answering**. That means it does not freely invent an answer. Instead, it:

- looks at the retrieved passages
- picks the most relevant snippets
- removes duplicates
- returns a concise grounded answer

This logic lives in [extractive.py](/D:/SIC/src/generation/extractive.py).

The project also supports an optional **hybrid** mode through config, where retrieved context can be sent to an external LLM for more natural answer writing.

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
- [extractive.py](/D:/SIC/src/generation/extractive.py): answer selection layer

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

### 5. Ask Questions

```bash
py -3.11 -m main query "What is machine learning?"
```

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
  - `max_context_tokens`

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
- giving grounded extractive answers
- learning custom semantic embeddings from the ingested corpus
- supporting local experimentation with different retrieval and embedding settings

## Limitations

This project is intentionally lighter and more classical than a full large language model stack, so it has a few important limitations:

- it is better at **retrieval and grounded extraction** than free-form generation
- answer quality depends heavily on chunk quality and retrieval quality
- very large technical PDFs can still produce imperfect ranking
- the Siamese model learns similarity, not full reasoning
- hybrid LLM mode requires an external API if enabled

## For Presentation or Viva

### One-Line Description

> This is a local RAG-style private document QA system that combines classical NLP, a custom Keras Siamese embedding model, hybrid retrieval, and extractive answering.

### Short Explanation

> The system reads PDFs, preprocesses and chunks their text, learns semantic embeddings for those chunks, retrieves the most relevant passages for a question, and then returns a grounded answer from the retrieved content.

### Academic Framing

You can describe it like this:

> This project combines classical NLP preprocessing with a neural Siamese embedding model for semantic retrieval over private documents. It uses hybrid retrieval by combining dense vector similarity and lexical matching, then performs extractive answer selection from the retrieved context.

### Why This Design Matters

- It is more grounded than free-form generation alone.
- It is more explainable because answers come from source documents.
- It supports private local document collections.
- It allows experimentation with retrieval and embedding models.

---

If you are presenting this project, the best summary is:

**"I built a local private-document QA system that uses custom embeddings and hybrid retrieval to answer questions directly from PDF content."**
