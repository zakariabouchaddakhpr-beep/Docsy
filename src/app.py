"""
app.py
Docsy — Chat with Supabase documentation.
A polished Streamlit chat interface with source citations.

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

# ---- Custom prompt (tuned for accuracy) -----------------------------------
QA_PROMPT = PromptTemplate(
    """\
You are Docsy, a friendly and knowledgeable AI assistant that answers questions \
about Supabase using ONLY the documentation excerpts provided below.

RULES:
1. Answer based ONLY on the provided context. Never make up information.
2. If the context doesn't contain enough information to fully answer the question, \
be honest: say "Based on the docs I have indexed, I don't have a complete answer \
for that." Then share whatever partial information IS relevant from the context.
3. Do NOT add a "Sources:" section — sources are displayed separately by the app.
4. Be concise but thorough. Use code examples from the docs when they help.
5. Format your answers with markdown for readability (headers, bold, code blocks).
6. If the user greets you or asks what you can do, introduce yourself as Docsy — \
an AI assistant that helps developers navigate Supabase documentation. Mention \
that you can answer questions about auth, database, storage, edge functions, \
and other Supabase features.
7. If the question is completely unrelated to Supabase or software development, \
politely redirect: "I'm specialized in Supabase documentation — I might not be \
the best help for that! Try asking me about auth, database, storage, or any \
other Supabase feature."

Context from Supabase documentation:
-----
{context_str}
-----

User question: {query_str}

Answer:"""
)

# ---- Starter questions (shown when chat is empty) -------------------------
STARTER_QUESTIONS = [
    "How do I set up email authentication?",
    "What is Row Level Security (RLS)?",
    "How do I upload files to Supabase Storage?",
    "What are Edge Functions?",
]

# ---- Keywords that indicate a real Supabase question ----------------------
SUPABASE_KEYWORDS = [
    "auth", "authentication", "login", "signup", "sign up", "sign in", "oauth",
    "password", "email", "mfa", "otp", "jwt", "token", "session", "user",
    "database", "table", "row", "column", "sql", "query", "insert", "select",
    "update", "delete", "rls", "row level security", "policy", "policies",
    "storage", "bucket", "upload", "file", "image", "download",
    "edge function", "function", "deno", "invoke", "deploy",
    "realtime", "subscribe", "channel", "broadcast", "presence",
    "supabase", "postgrest", "postgres", "api", "key", "anon", "service role",
    "migration", "schema", "index", "vector", "embedding", "search",
    "webhook", "trigger", "cron", "queue", "self-host", "cli",
    "react", "next", "flutter", "swift", "kotlin", "python", "javascript",
]
# ---------------------------------------------------------------------------

# ---- Page config ----------------------------------------------------------
st.set_page_config(
    page_title="Docsy — Supabase Docs Assistant",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ---- Custom CSS -----------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=JetBrains+Mono:wght@400;500&display=swap');

    .stApp {
        font-family: 'DM Sans', sans-serif;
    }

    /* Header */
    .docsy-header {
        text-align: center;
        padding: 1.5rem 0 0.5rem 0;
    }
    .docsy-header h1 {
        font-family: 'DM Sans', sans-serif;
        font-weight: 700;
        font-size: 2.2rem;
        margin-bottom: 0.25rem;
        letter-spacing: -0.03em;
    }
    .docsy-header .tagline {
        opacity: 0.55;
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

    .source-divider {
        font-size: 0.75rem;
        opacity: 0.5;
        margin-top: 12px;
        margin-bottom: 6px;
    }

    /* Stats */
    .stats-badge {
        text-align: center;
        padding: 4px 0 8px 0;
        font-size: 0.75rem;
        opacity: 0.35;
    }

    /* Starter question buttons */
    .stButton > button {
        border-radius: 20px !important;
        font-size: 0.85rem !important;
        padding: 0.4rem 1rem !important;
        font-family: 'DM Sans', sans-serif !important;
    }

    /* Hide Streamlit chrome */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def load_index() -> tuple[VectorStoreIndex, int]:
    """Load ChromaDB index. Cached — runs once per session."""

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error(
            "**GROQ_API_KEY not found.** "
            "Create a `.env` file in the project root with your free key from "
            "[console.groq.com](https://console.groq.com)."
        )
        st.stop()

    if not CHROMA_DIR.exists():
        st.error(
            "**Knowledge base not found.** "
            "Run `python src/build_index.py` to build it first."
        )
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


def is_supabase_question(question: str) -> bool:
    """Check if the question is actually about Supabase / technical topics."""
    q = question.lower()
    return any(kw in q for kw in SUPABASE_KEYWORDS)


def query_docsy(index: VectorStoreIndex, question: str) -> tuple[str, list[str]]:
    """Ask a question, return (answer, source_urls)."""
    query_engine = index.as_query_engine(
        similarity_top_k=5,
        text_qa_template=QA_PROMPT,
    )

    try:
        response = query_engine.query(question)
    except Exception as e:
        error_msg = str(e).lower()
        if "rate" in error_msg or "limit" in error_msg or "429" in error_msg:
            return (
                "⏳ I'm being rate-limited by the AI provider. "
                "Please wait about 30 seconds and try again.",
                [],
            )
        elif "auth" in error_msg or "key" in error_msg or "401" in error_msg:
            return (
                "🔑 There seems to be an issue with the API key. "
                "Please check your `.env` file.",
                [],
            )
        else:
            return (
                f"Something went wrong while generating the answer. "
                f"Please try again.\n\n*Error: {e}*",
                [],
            )

    answer = str(response).strip()

    # Only show sources if the question is actually about Supabase
    if is_supabase_question(question):
        sources = list(
            dict.fromkeys(
                node.metadata.get("source_url", "")
                for node in response.source_nodes
                if node.metadata.get("source_url")
            )
        )
    else:
        sources = []

    return answer, sources


def render_sources(sources: list[str]):
    """Render source URLs as clickable chips."""
    if not sources:
        return

    st.markdown('<div class="source-divider">📚 Sources</div>', unsafe_allow_html=True)

    chips_html = ""
    for url in sources:
        label = url.split("/docs/")[-1] if "/docs/" in url else url
        label = label.strip("/").replace("/", " › ")
        chips_html += f'<a href="{url}" target="_blank" class="source-chip">{label}</a>'

    st.markdown(chips_html, unsafe_allow_html=True)


# ---- Sidebar (always visible) --------------------------------------------
with st.sidebar:
    st.markdown("## 📚 Docsy")
    st.caption("Your docs, but they actually answer back.")

    st.divider()

    if st.button("🗑️ New chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_question = None
        st.rerun()

    st.divider()

    st.markdown(
        """
        **Tech stack**

        🦙 Llama 3.3 70B via Groq
        🔍 LlamaIndex RAG
        💾 ChromaDB
        🖥️ Streamlit

        ---

        [Supabase Docs](https://supabase.com/docs)
        """,
    )

# ---- Main app -------------------------------------------------------------

# Header
st.markdown(
    """
    <div class="docsy-header">
        <h1>📚 Docsy</h1>
        <p class="tagline">Your docs, but they actually answer back.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Load index
with st.spinner("Loading knowledge base..."):
    index, chunk_count = load_index()

st.markdown(
    f'<div class="stats-badge">Llama 3.3 · {chunk_count:,} chunks indexed from Supabase docs</div>',
    unsafe_allow_html=True,
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

# Show starter questions if chat is empty
if not st.session_state.messages and st.session_state.pending_question is None:
    st.markdown("")  # spacing
    cols = st.columns(2)
    for i, question in enumerate(STARTER_QUESTIONS):
        with cols[i % 2]:
            if st.button(question, key=f"starter_{i}", use_container_width=True):
                st.session_state.pending_question = question
                st.rerun()

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            render_sources(msg["sources"])

# Handle pending question from starter buttons
if st.session_state.pending_question is not None:
    question = st.session_state.pending_question
    st.session_state.pending_question = None

    # Show user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, sources = query_docsy(index, question)
        st.markdown(answer)
        render_sources(sources)

    # Save to history
    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )

# Chat input
if prompt := st.chat_input("Ask anything about Supabase..."):
    # Add user message
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