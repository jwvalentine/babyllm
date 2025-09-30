# BabyLLM

**BabyLLM** is a lightweight, locally-hosted Retrieval-Augmented Generation (RAG) service for teams who want private, on‑prem AI. It lets you ingest internal knowledge (Markdown, text, PDFs, Word/Excel, CSVs, etc.), index it with **Hugging Face embeddings** (local‑only), store vectors in **ChromaDB**, and ask questions via a simple REST API. Answers include citations to the most relevant source chunks.

> **Project context:** BabyLLM uses the broader **Ollama** project and intentionally uses **Hugging Face for embeddings only** (no Ollama embeddings). Inference is handled by **Ollama**. The system is designed to run **entirely offline** on your own hardware.

---

## Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Quickstart](#-quickstart)
- [API](#-api)
- [Deployment Options](#-deployment-options)
  - [Docker Compose (recommended)](#docker-compose-recommended)
  - [Bare-Metal (Python)](#bare-metal-python)
- [Configuration](#-configuration)
  - [.env.example](#envexample)
- [Ingestion Notes (PDF/OCR)](#ingestion-notes-pdfocr)
- [Operations & Maintenance](#-operations--maintenance)
- [Troubleshooting](#-troubleshooting)
- [License](#-license)
- [Example Workflow](#-example-workflow)
- [Repo Structure](#-repo-structure)

---

## 🚀 Features

- **Document ingestion API** for `.md`, `.txt`, `.pdf`, `.docx`, `.xlsx`, `.csv`, and more.
- **Local embeddings** with Hugging Face (e.g., `sentence-transformers/all-MiniLM-L6-v2`), stored in **ChromaDB**.
- **Semantic retrieval** + **context-aware generation** via a local Ollama model (e.g., `mistral`, `llama3`, etc.).
- **Citations**: responses include the file and chunk used to answer.
- **Stateless API** with a **persistent vector store** (on disk or Chroma server).
- **CPU‑first**; optional GPU acceleration when available.
- **Local‑only by design** — your data never leaves your machine.

---

## 🧩 Architecture

```
[ Files ] ──► [Chunker] ──► [HF Embeddings] ──► [ChromaDB]
                                   ▲
                                   │ (retrieve top‑k)
                                   ▼
                               [Retriever] ──► [Prompt Builder] ──► [Ollama LLM] ──► Answer + Citations
```

- **Chunker**: splits docs (e.g., 1000 tokens, 200 overlap).
- **Embeddings**: Hugging Face model runs locally on CPU/GPU.
- **Store**: ChromaDB persists vectors (embedded or server mode).
- **Generation**: Retrieved chunks → prompt → local Ollama for grounded answers.

---

## ⚡ Quickstart

1) **Run services** (see [Docker Compose](#docker-compose-recommended) or [Bare‑Metal](#bare-metal-python)).  
2) **Ingest a file**:
```bash
curl -X POST http://localhost:7209/api/ingest   -F "files=@README.md"
```
Example response:
```json
{"added":1,"fake":false}
```
3) **Ask a question**:
```bash
curl -X POST http://localhost:7209/api/ask   -H "Content-Type: application/json"   -d '{"question":"What database does BabyLLM use?"}'
```
Example response:
```json
{
  "answer": "ChromaDB",
  "sources": [
    { "chunk": 0, "source": "README.md" }
  ]
}
```

---

## 📡 API

### `POST /api/ingest`
- **FormData**: one or more `files=@path/to/file`
- **Effect**: chunk → HF embed → upsert into ChromaDB
- **Response**:
```json
{"added": <count>, "fake": false}
```
**Notes**
- Re‑posting a file re‑embeds and updates the store.
- Scanned PDFs require OCR; see [PDF/OCR](#ingestion-notes-pdfocr).

### `POST /api/ask`
- **Body**:
```json
{ "question": "Your question here", "k": 4 }
```
- `k` (optional): number of chunks to retrieve (default 4).  
- **Response**:
```json
{
  "answer": "string",
  "sources": [
    { "chunk": 12, "source": "docs/guide.md" }
  ]
}
```
**Notes**
- Answers are grounded in retrieved context; tune `k` for breadth vs. precision.

---

## 🧪 Deployment Options

### Docker Compose (recommended)

Create `docker-compose.yml` in your repo:

```yaml
version: "3.9"
services:
  ollama:
    image: ollama/ollama:latest
    restart: unless-stopped
    ports:
      - "11434:11434"
    volumes:
      - ollama_models:/root/.ollama
    environment:
      - OLLAMA_HOST=0.0.0.0

  # Optional: run Chroma as a server (recommended for multi-container setups).
  chroma:
    image: chromadb/chroma:latest
    restart: unless-stopped
    ports:
      - "8000:8000"
    volumes:
      - ./data/chroma:/chroma
    environment:
      - IS_PERSISTENT=TRUE

  babyllm:
    build: .
    restart: unless-stopped
    depends_on:
      - ollama
      - chroma
    env_file:
      - .env
    ports:
      - "7209:7209"
    volumes:
      - ./data:/data
      - ./ingest:/ingest

volumes:
  ollama_models:
```

Then run:
```bash
# 1) Start everything
docker compose up -d --build

# 2) (One-time) Pull an Ollama model you want to use:
docker exec -it $(docker ps -qf "name=ollama") ollama pull mistral

# 3) Ingest and ask (see Quickstart)
```

> **Embedded vs Server Chroma:** If you prefer embedded Chroma (no separate container), omit the `chroma` service and set `CHROMA_PERSIST_DIR` in `.env`. For server mode, set `CHROMA_HOST=http://chroma:8000`.

---

### Bare‑Metal (Python)

```bash
# System deps (example: Ubuntu)
sudo apt-get update && sudo apt-get install -y python3 python3-venv python3-pip

# Project
git clone https://github.com/your-org/babyllm.git
cd babyllm

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env to choose your HF embedding model and Ollama model

# Run API (FastAPI/Uvicorn)
uvicorn app.main:app --host 0.0.0.0 --port 7209
```

Optional: run **Ollama** locally:
```bash
# Install Ollama from https://github.com/ollama/ollama
ollama serve
ollama pull phi3:mini
```

---

## 🧷 Configuration

BabyLLM is configured via environment variables. Common knobs:

- **Embeddings**
  - `EMBEDDING_MODEL` — HF model ID (e.g., `sentence-transformers/all-MiniLM-L6-v2`)
  - `EMBEDDING_DEVICE` — `cpu` | `cuda` | `mps`
  - `CHUNK_SIZE` / `CHUNK_OVERLAP` — chunking strategy
- **Vector Store**
  - Embedded: `CHROMA_PERSIST_DIR=/data/chroma`
  - Server: `CHROMA_HOST=http://chroma:8000`
- **Ollama LLM**
  - `OLLAMA_HOST=http://ollama:11434`
  - `OLLAMA_MODEL=mistral` (or `llama3`, etc.)
  - `MAX_CONTEXT_TOKENS=4096` (depends on model)
- **API**
  - `APP_PORT=7209`
  - `ALLOWED_ORIGINS=*`
- **OCR (optional)**
  - `ENABLE_PDF_OCR=false`
  - `EASYOCR_LANGS=en`

### `.env.example`

```env
# ── App ────────────────────────────────────────────────────────────────────────
APP_PORT=7209
ALLOWED_ORIGINS=*

# ── Embeddings (HF only; no Ollama embeddings) ────────────────────────────────
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DEVICE=cpu
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# ── ChromaDB (choose one mode) ────────────────────────────────────────────────
# Embedded (on-disk)
CHROMA_PERSIST_DIR=/data/chroma
# Server mode (comment the persist dir above and use server host instead)
# CHROMA_HOST=http://chroma:8000

# ── Ollama (LLM inference) ────────────────────────────────────────────────────
OLLAMA_HOST=http://ollama:11434
OLLAMA_MODEL=mistral
MAX_CONTEXT_TOKENS=4096

# ── OCR (optional; PDFs with images) ──────────────────────────────────────────
ENABLE_PDF_OCR=false
EASYOCR_LANGS=en
```

---

## 📄 Ingestion Notes (PDF/OCR)

- **Text‑based PDFs**: parsed directly.
- **Scanned PDFs** (image‑only): set `ENABLE_PDF_OCR=true` to run local OCR via **EasyOCR** (Apache‑2.0). This keeps processing offline but may be slower on CPU. You can specify multiple languages in `EASYOCR_LANGS` (e.g., `en,es`).

> If OCR is disabled, scanned PDFs will ingest with minimal/empty text.

---

## 🛠 Operations & Maintenance

- **Backups**: back up your Chroma data directory (embedded: `CHROMA_PERSIST_DIR`; server: the mounted volume).
- **Re‑ingest**: re‑POST files to `/api/ingest` after content changes.
- **Model swaps**: you can switch `OLLAMA_MODEL`; re‑evaluate `MAX_CONTEXT_TOKENS` accordingly.
- **Performance tips**:
  - CPU‑only works; **32GB RAM** recommended for larger corpora.
  - Use smaller embedding models for speed; use GPU (`EMBEDDING_DEVICE=cuda`) when available.
  - Tune retrieval `k` in `/api/ask` (typical range 3–8).

---

## 🧰 Troubleshooting

- **No model loaded / slow first token**: ensure `ollama serve` is running and you’ve pulled the model (`ollama pull mistral`). First response can be slower due to model load/warmup.
- **Empty answers**: confirm files contain text (or enable OCR for scanned PDFs). Increase `k` in `/api/ask`.
- **Vector store not persisting**: check that `CHROMA_PERSIST_DIR` exists and is writable (embedded), or that `CHROMA_HOST` is reachable (server).
- **CORS errors**: set `ALLOWED_ORIGINS=*` (or your domain) when calling from a browser app.

---

## 🛡 License

**MIT** — see `LICENSE`. Core dependencies used by BabyLLM are permissive (MIT/Apache‑2.0). Hugging Face models and EasyOCR are local and compatible with closed‑source deployment; review each model’s license before distribution.

---

## 🙌 Example Workflow

1. Ingest your internal docs:
   ```bash
   curl -X POST http://localhost:7209/api/ingest      -F "files=@docs/runbook.md"      -F "files=@handbook.pdf"      -F "files=@tickets.csv"
   ```
2. Ask a task‑oriented question:
   ```bash
   curl -X POST http://localhost:7209/api/ask      -H "Content-Type: application/json"      -d '{"question":"How do we rotate the production API keys?", "k": 6}'
   ```
3. Use the cited sources to verify/trace the answer.

---

## 📁 Repo Structure

```
babyllm/
├─ app/
│  ├─ main.py              # FastAPI app (routes: /api/ingest, /api/ask)
│  ├─ ingest.py            # Chunking, file parsing, optional OCR
│  ├─ embeddings.py        # HF embedding pipeline
│  ├─ store.py             # ChromaDB (embedded/server client)
│  └─ rag.py               # Retriever + prompt builder
├─ requirements.txt
├─ docker-compose.yml
├─ Dockerfile
├─ .env.example
└─ README.md
```
