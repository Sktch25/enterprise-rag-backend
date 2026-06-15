# Enterprise RAG API

A production-ready Retrieval-Augmented Generation (RAG) backend built with Python, FastAPI, and LangChain. This API ingests document data, generates vector embeddings, stores them locally, and uses Google's Gemini LLMs to answer queries with strict, hallucination-free grounding.

## Tech Stack
* **Framework:** FastAPI
* **AI Orchestration:** LangChain
* **Vector Database:** ChromaDB
* **LLM & Embeddings:** Google Gemini API (`gemini-3.5-flash` & `gemini-embedding-001`)
* **Document Parsing:** PyMuPDF (fitz)

## Architecture Overview
1. **Ingestion (`build_db.py`):** Parses PDF documents, splits text into semantic chunks, and generates vector embeddings via the Gemini API.
2. **Storage:** Embeddings are persistently saved to a local Chroma SQLite database to avoid redundant API calls.
3. **Retrieval (`query_db.py`):** Semantically searches the local vector store for the top contextual matches.
4. **API (`main.py`):** Exposes the RAG pipeline via a high-performance FastAPI endpoint.

## Local Setup & Installation

**1. Clone the repository**
```bash
git clone [https://github.com/YourUsername/enterprise-rag-backend.git](https://github.com/YourUsername/enterprise-rag-backend.git)
cd enterprise-rag-backend