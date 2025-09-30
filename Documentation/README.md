# BabyLLM

BabyLLM is a lightweight Retrieval Augmented Generation (RAG) system.

- **Ollama** provides embeddings and large language model inference.
- **ChromaDB** stores vector embeddings for semantic search.
- **.NET 8** hosts the API with endpoints for ingestion and question answering.

### Workflow
1. Upload your documents via `/api/ingest`.
2. BabyLLM chunks the text, embeds with Ollama, and stores in Chroma.
3. Ask questions via `/api/ask` — BabyLLM retrieves context from Chroma and generates an answer.
