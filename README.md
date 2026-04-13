# Docsy 📚

> Your docs, but they actually answer back.

Docsy is an AI assistant that turns any product's documentation into a conversation. Instead of digging through pages of docs to find one answer, you just ask — and Docsy responds with a clear answer plus links to the exact sources it used.

Built with **Llama 3.3**, **LlamaIndex**, **ChromaDB**, and **Streamlit**.

> 🚧 **Status:** Day 1 of 7 — project setup + documentation scraper complete.

---

## How it works

Docsy uses Retrieval-Augmented Generation (RAG):

1. **Ingest** — documentation pages are scraped and saved as plain text.
2. **Embed** — pages are split into chunks and converted into vector embeddings.
3. **Store** — embeddings are stored locally in ChromaDB.
4. **Retrieve** — when a user asks a question, the most relevant chunks are pulled.
5. **Generate** — Llama 3.3 (via Groq) generates a grounded answer with citations.

## Tech stack

| Layer | Tool |
|---|---|
| LLM | Llama 3.3 70B via [Groq](https://console.groq.com) (free) |
| Embeddings | `BAAI/bge-small-en-v1.5` (runs locally, free) |
| RAG framework | LlamaIndex |
| Vector DB | ChromaDB (local) |
| UI | Streamlit |
| Deployment | Hugging Face Spaces |

## Setup

```bash
# 1. Clone and enter the repo
git clone https://github.com/YOUR_USERNAME/docsy.git
cd docsy

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate    # on Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your free Groq API key
cp .env.example .env
# then edit .env and paste your key from https://console.groq.com

# 5. Test the connection
python src/hello_groq.py

# 6. Scrape some docs
python src/scrape_docs.py
```

## Roadmap

- [x] Day 1 — Project setup, Groq connection, docs scraper
- [ ] Day 2 — RAG pipeline (embed + store + retrieve)
- [ ] Day 3 — Streamlit chat UI with source citations
- [ ] Day 4 — Polish, edge cases, error handling
- [ ] Day 5 — Deploy to Hugging Face Spaces
- [ ] Day 6 — Architecture diagram, screenshots, full README
- [ ] Day 7 — Demo video + LinkedIn launch

## About

Built as a portfolio project to demonstrate end-to-end RAG system design — from data ingestion to deployed product.
