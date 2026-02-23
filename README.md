# 🧠 Profile Chat — RAG-Powered Document Q&A

A locally-running, privacy-first chatbot that lets you upload any personal profile document and ask natural-language questions about it. Powered by **LangChain**, **ChromaDB**, **Ollama**, and **Streamlit** — no cloud API keys required.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running the App](#running-the-app)
- [Configuration](#configuration)
- [Usage Guide](#usage-guide)
- [Module Reference](#module-reference)
- [Customization](#customization)
- [Troubleshooting](#troubleshooting)

---

## Overview

Profile Chat uses **Retrieval-Augmented Generation (RAG)** to answer questions about a person based on a document you provide. Instead of sending your data to a cloud service, everything runs locally on your machine — the embedding model, the vector database, and the LLM all run offline.

The intended use case is a scalable, reusable chatbot that works for **any person's profile** — just upload a new document and the system adapts automatically, including extracting the person's name.

---

## Features

- 📄 **Upload any profile document** — `.txt` or `.md` files supported
- 👤 **Auto name detection** — extracts the person's name from the document automatically
- ✂️ **Smart chunking** — Markdown-aware splitting with recursive fallback for long sections
- 🔢 **Local embeddings** — uses `sentence-transformers/all-MiniLM-L6-v2` via HuggingFace, runs on CPU
- 🗄️ **Persistent vector store** — ChromaDB persists embeddings to disk; re-initialization is fast on subsequent runs
- 🤖 **Local LLM** — powered by Ollama (`mistral:latest` by default, easily swappable)
- 💬 **Clean chat UI** — conversation-style interface built with Streamlit
- ↩️ **Re-upload anytime** — switch to a different document without restarting the app
- 🔒 **Fully offline** — no data leaves your machine

---

## Project Structure

```
My Profile using RAG/
│
├── assets/
│   ├── my_profile.txt          # Default profile document (fallback)
│   └── chroma_db/              # ChromaDB vector store (auto-created)
│
├── src/
│   ├── config.py               # Central configuration (paths, models, chunk settings)
│   ├── preprocessing_data.py   # Document loading and smart chunking
│   ├── vectorstore.py          # Embedding model and ChromaDB management
│   ├── model_loader.py         # Ollama LLM singleton loader
│   ├── rag_pipeline.py         # Retriever setup and RAG answer function
│   │
│   └── views/
│       ├── __init__.py
│       ├── streamlit_app.py    # Main Streamlit application
│       └── components.py       # Reusable UI components (legacy)
│
├── requirements.txt
└── README.md
```

---

## How It Works

The pipeline follows a standard RAG architecture with one pass at initialization time and fast retrieval at query time.

```
Upload Document
      │
      ▼
 Load & Parse          ← TextLoader reads the .txt / .md file
      │
      ▼
 Smart Chunking        ← MarkdownHeaderTextSplitter splits by # headers
      │                   RecursiveCharacterTextSplitter handles long sections
      ▼
 Embed Chunks          ← sentence-transformers/all-MiniLM-L6-v2 (local, CPU)
      │
      ▼
 Store in ChromaDB     ← persisted to disk; reused on next run
      │
      ▼
──── Ready to Chat ────

User Query
      │
      ▼
 Similarity Search     ← top-k=5 most relevant chunks retrieved
      │
      ▼
 Build Prompt          ← name + context + question injected into PromptTemplate
      │
      ▼
 Ollama LLM            ← mistral:latest generates a natural answer
      │
      ▼
 Formatted Response    ← emoji prefix added based on response type
```

---

## Prerequisites

Make sure the following are installed on your machine before proceeding.

**Python 3.10+**
Verify with `python --version`.

**Ollama**
Download from [https://ollama.com](https://ollama.com) and install for your OS. After installing, pull the model:

```bash
ollama pull mistral
```

To verify Ollama is running:

```bash
ollama list
```

You should see `mistral:latest` in the list.

---

## Installation

**1. Clone or download the project**

```bash
git clone <https://github.com/asmaa-2ahmed/explore_your_document>
cd "explore_your_document"
```

**2. Create and activate a virtual environment**

```bash
# Windows
python -m venv Rag_env
Rag_env\Scripts\activate

# macOS / Linux
python -m venv Rag_env
source Rag_env/bin/activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

---

## Running the App

Make sure your virtual environment is active and Ollama is running, then launch:

```bash
streamlit run src/views/streamlit_app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

> **Note:** The first run will download the `all-MiniLM-L6-v2` embedding model (~90 MB) from HuggingFace. This only happens once; subsequent runs use the cached model.

---

## Configuration

All configuration lives in `src/config.py`. Edit this file to change models, paths, or chunk settings.

| Setting | Default | Description |
|---|---|---|
| `OLLAMA_MODEL` | `mistral:latest` | The Ollama model used for generation. Swap to `gemma3:4b`, `llama3`, etc. |
| `EMBEDDINGS_MODEL_ID` | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace embedding model |
| `CHROMA_DB_PATH` | `assets/chroma_db` | Where ChromaDB persists its vector store |
| `file_path` | `assets/my_profile.txt` | Default fallback document if no file is uploaded |
| `CHUNK_SIZE` | `1200` | Max characters per chunk |
| `CHUNK_OVERLAP` | `150` | Overlap between adjacent chunks to preserve context |
| `DEVICE` | `cpu` | Device for embeddings (`cpu` or `cuda`) |

**Switching LLM models**

Edit `config.py`:

```python
OLLAMA_MODEL = "gemma3:4b"   # or llama3, phi3, etc.
```

Then pull the model with Ollama before running:

```bash
ollama pull gemma3:4b
```

---

## Usage Guide

The app walks you through three stages.

**Stage 1 — Upload**

Drag and drop or click to upload your `.txt` or `.md` profile file. The app will immediately scan the document and attempt to detect the person's name. Review the detected name in the text field and correct it if needed, then click **Continue**.

**Stage 2 — Initialize**

Click **Build & Start Chatting**. You'll see a live progress display:

- 📄 Load document — reads and parses the file
- ✂️ Split into chunks — applies smart Markdown-aware splitting
- 🔢 Build vector store — embeds chunks and stores them in ChromaDB
- 🔍 Set up retriever — configures similarity search

This step takes 15–60 seconds on first run (embedding time). On subsequent runs for the same file, ChromaDB loads from disk and this is nearly instant.

**Stage 3 — Chat**

Type any question about the person in the chat input at the bottom. The system will retrieve the most relevant sections of the document and use the LLM to generate a natural, paraphrased answer.

Example questions you might ask:
- *"What is their educational background?"*
- *"What programming languages do they know?"*
- *"Tell me about their work experience."*
- *"What projects have they worked on?"*

If the answer isn't in the document, the model will say so rather than guessing.

**Uploading a new document**

Click **↩ New File** at any time to return to the upload stage. The previous document's data and chat history are cleared automatically.

## Customization

**Change the number of retrieved chunks**

In `src/rag_pipeline.py`, adjust the default `k` value:

```python
def get_retriever(vectorstore, k=5):   # increase for more context
```

Or modify it in `utils.py` when calling `get_retriever`.

**Modify the prompt**

Edit the `PROMPT` template in `src/rag_pipeline.py` to change how the LLM is instructed to respond:

```python
PROMPT = PromptTemplate.from_template("""
You are an AI assistant answering questions about {name}.
...
""")
```

**Change chunk size**

If answers feel cut off or lack context, increase `CHUNK_SIZE` in `config.py`. If retrieval feels too broad, decrease it.

```python
CHUNK_SIZE = 1500    # larger chunks = more context per retrieved piece
CHUNK_OVERLAP = 200  # larger overlap = less chance of missing cross-boundary info
```
---

**Tech Stack**

| Component | Library / Tool |
|---|---|
| UI | Streamlit |
| Orchestration | LangChain |
| Embeddings | sentence-transformers / HuggingFace |
| Vector Store | ChromaDB |
| LLM | Ollama (Mistral, Gemma, LLaMA, etc.) |
| Text Splitting | LangChain MarkdownHeaderTextSplitter + RecursiveCharacterTextSplitter |