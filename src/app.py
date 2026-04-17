"""
app.py
Docsy — Chat with Supabase documentation.
A clean Streamlit chat interface with source citations.

Usage:
    streamlit run src/app.py
"""
import os
from pathlib import Path

import chromadb
import streamlit as st
from dotenv import load_dotenv
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.prompts import PromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.vector_stores.chroma import ChromaVectorStore

load_dotenv()

# ---- Paths ----------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "docsy_docs"
# ---------------------------------------------------------------------------

# ---- Custom prompt --------------------------------------------------------
QA_PROMPT = PromptTemplate(
    """\
You are Docsy, a helpful AI assistant that answers questions about Supabase \
using ONLY the documentation excerpts provided below.

RULES:
1. Answer based ONLY on the provided context. If the context doesn't contain \
enough information, say "I don't have enough information in the docs to answer \
that fully" and share what you DO know.
2. Do NOT add a "Sources:" section in your answer — sources are handled separately.
3. Be concise but thorough. Use code examples from the docs when relevant.
4. Format your answers with markdown for readability.
5. If the user greets you or asks what you can do, introduce yourself as Docsy, \
an AI assistant for Supabase documentation.

Context from Supabase documentation:
-----
{context_str}
-----

User question: {query_str}

Answer:"""
)
# ---------------------------------------------------------------------------

# ---- Page config ----------------------------------------------------------
st.set_page_config(
    page_title="Docsy — Supabase Docs Assistant",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ---- Custom CSS -----------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=JetBrains+Mono:wght@400;500&display=swap');

    /* Global */
    .stApp {
        font-family: 'DM Sans', sans-serif;
    }

    /* Header area */
    .docsy-header {
        text-align: center;
        padding: 1.5rem 0 1rem 0;
    }
    .docsy-header h1 {
        font-family: 'DM Sans', sans-serif;
        font-weight: 700;
        font-size: 2rem;
        margin-bottom: 0.25rem;
        letter-spacing: -0.02em;
    }
    .docsy-header p {
        opacity: 0.6;
        font-size: 0.95rem;
        margin: 0;
    }

    /* Source chips */
    .source-chip {
        display: inline-block;
        padding: 4px 12px;
        margin: 3px 4px 3px 0;
        border-radius: 20px;
        font-size: 0.78rem;
        font-family: 'JetBrains Mono', monospace;
        background: rgba(59, 130, 246, 0.1);
        color: rgb(59, 130, 246);
        text-decoration: none;
        border: 1px solid rgba(59, 130, 246, 0.2);
        transition: all 0.2s ease;
    }
    .source-chip:hover {
        background: rgba(59, 130, 246, 0.2);
        border-color: rgba(59, 130, 246, 0.4);
        text-decoration: none;
    }

    /* Subtle divider */
    .source-divider {
        font-size: 0.75rem;
        opacity: 0.5;
        margin-top: 12px;
        margin-bottom: 6px;
    }

    /* Stats badge */
    .stats-badge {
        text-align: center;
        padding: 6px 0;
        font-size: 0.75rem;
        opacity: 0.4;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def load_index() -> tuple[VectorStoreIndex, int]:
    """Load ChromaDB index (cached so it only runs once per session)."""

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("GROQ_API_KEY not found. Add it to your .env file.")
        st.stop()

    if not CHROMA_DIR.exists():
        st.error("ChromaDB not found. Run `python src/build_index.py` first.")
        st.stop()

    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.llm = Groq(
        model="llama-3.3-70b-versatile",
        api_key=api_key,
        temperature=0.1,
    )

    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    chroma_collection = chroma_client.get_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    index = VectorStoreIndex.from_vector_store(vector_store)

    return index, chroma_collection.count()


def query_docsy(index: VectorStoreIndex, question: str) -> tuple[str, list[str]]:
    """Ask a question, return (answer, list_of_source_urls)."""
    query_engine = index.as_query_engine(
        similarity_top_k=5,
        text_qa_template=QA_PROMPT,
    )

    response = query_engine.query(question)

    # Extract unique source URLs
    sources = list(
        dict.fromkeys(
            node.metadata.get("source_url", "")
            for node in response.source_nodes
            if node.metadata.get("source_url")
        )
    )

    return str(response).strip(), sources


def render_sources(sources: list[str]):
    """Render source URLs as clickable chips."""
    if not sources:
        return

    st.markdown('<div class="source-divider">📚 Sources</div>', unsafe_allow_html=True)

    chips_html = ""
    for url in sources:
        # Extract a readable label from the URL
        label = url.split("/docs/")[-1] if "/docs/" in url else url
        label = label.strip("/").replace("/", " › ")
        chips_html += f'<a href="{url}" target="_blank" class="source-chip">{label}</a>'

    st.markdown(chips_html, unsafe_allow_html=True)


# ---- Main app -------------------------------------------------------------

# Header
st.markdown(
    """
    <div class="docsy-header">
        <h1>📚 Docsy</h1>
        <p>Your docs, but they actually answer back.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Load index
with st.spinner("Loading knowledge base..."):
    index, chunk_count = load_index()

# Stats
st.markdown(
    f'<div class="stats-badge">Llama 3.3 · {chunk_count} chunks indexed from Supabase docs</div>',
    unsafe_allow_html=True,
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            render_sources(msg["sources"])

# Chat input
if prompt := st.chat_input("Ask anything about Supabase..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, sources = query_docsy(index, prompt)

        st.markdown(answer)
        render_sources(sources)

    # Save to history
    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
