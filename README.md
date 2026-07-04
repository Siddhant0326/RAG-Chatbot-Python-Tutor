# 🐍 Python Tutor — RAG-based Personalized Education System

A Retrieval-Augmented Generation (RAG) chatbot that answers Python questions
grounded in a source PDF (e.g. a Python handbook), built with LangChain,
FAISS, HuggingFace embeddings, Groq (Llama 3.1), and Streamlit.

> **Status:** Learning / prototype project. Core RAG pipeline and guardrails
> are implemented; not yet hardened for production use (see Roadmap below).

## Features

- PDF ingestion and chunking (`langchain` document loaders + text splitters)
- Local embeddings via HuggingFace (`sentence-transformers/all-MiniLM-L6-v2`)
- FAISS vector store for semantic retrieval, cached to disk after first build
- Two hallucination guardrails:
  1. **Distance threshold** — rejects questions with no relevant match in the document
  2. **LLM-as-judge coverage check** — a second LLM call verifies retrieved
     context actually contains enough info before generating an answer
- Multi-chat Streamlit UI (new chat, chat history, delete chat)
- Personalization hooks (`student_level`, `weak_area`)

## How it works

1. `load_documents` — loads the PDF via `PyPDFLoader`
2. `split_documents` — splits into ~500-character overlapping chunks
3. `create_vector_store` — embeds chunks and builds/loads a FAISS index
4. `create_rag_components` — sets up the Groq LLM and prompt template
5. `ask_question` — retrieves relevant chunks, runs guardrails, then
   generates a grounded answer

## Setup

```bash
git clone <this-repo>
cd <this-repo>
pip install -r requirements.txt
```

1. Copy `.env.example` to `.env` and add your `GROQ_API_KEY`
   (get one at https://console.groq.com)
2. Place your source PDF in the project root as `python_handbook.pdf`,
   or set `PDF_PATH` in `.env` to point elsewhere
3. Run the app:

```bash
streamlit run app.py
```

On first run, the app will build and cache a FAISS index (`faiss_index/`)
from the PDF. Subsequent runs load the cached index and start faster.

## Configuration

| Env var | Required | Default | Purpose |
|---|---|---|---|
| `GROQ_API_KEY` | Yes | — | Groq API key for the LLM |
| `PDF_PATH` | No | `python_handbook.pdf` | Path to the source PDF |
| `FAISS_INDEX_PATH` | No | `faiss_index` | Where the vector index is cached |

## Known limitations

This is a prototype-stage project. Notable gaps:

- Chat history is stored only in Streamlit session state and is lost on
  refresh/restart (no database)
- Single document / single tenant — no multi-user isolation or auth
- No rate limiting — a public deployment could exhaust API quota
- No automated evaluation of retrieval/answer quality
- Basic `print()`-based logging, no observability/tracing

## Roadmap (prototype → MVP)

- [ ] Persist chat history (SQLite)
- [ ] Basic per-session usage limits
- [ ] Deploy to Streamlit Community Cloud / Render
- [ ] Clearer user-facing error states (in progress — see `main.py`)

## Tech stack

LangChain · FAISS · HuggingFace `sentence-transformers` · Groq (Llama 3.1 8B) · Streamlit
