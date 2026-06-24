# Enterprise RAG Backend API

An enterprise-grade Retrieval-Augmented Generation (RAG) backend engine designed to ingest corporate documents, process them into high-density vector embeddings, and provide context-aware answers to user queries using FastAPI, LangChain, and the Google Gemini API.

##  Features

* **Asynchronous FastAPI Architecture:** Fast, production-ready REST API layout featuring seamless endpoint endpoints for document uploads and user queries.
* **Dynamic Directory Ingestion:** Automatical monitoring of the `data/` directory to load, parse, and split any uploaded PDF files on-the-fly.
* **Production-Grade Rate-Limit Handling:** Tailored document-chunk batching infrastructure that respects the Gemini free tier (`100 RPM`) by auto-throttling requests during deep embedding builds.
* **Local Vector Store Persistence:** Utilizes ChromaDB to physically cache token embeddings locally, drastically minimizing repetitive API processing costs.
* **Hallucination Prevention:** Strict prompt engineering ensures the engine answers strictly within the context of ingested documents or clearly states its lack of supporting facts.

---

##  Tech Stack

* **Framework:** FastAPI
* **Server:** Uvicorn
* **Orchestration:** LangChain
* **Vector Database:** ChromaDB (via `langchain-chroma`)
* **Embeddings Model:** `models/gemini-embedding-001`
* **LLM Engine:** Google Gemini API

---

##  Project Structure

```text
enterprise-rag-backend/
├── .env                  # Environment secrets (Gemini API Key)
├── .gitignore            # Version control filters
├── build_db.py           # Core ingestion loop & text chunking pipeline
├── main.py               # FastAPI router and server initialization
├── query_db.py           # RAG retrieval and pipeline matching logic
└── requirements.txt      # Project library dependencies
