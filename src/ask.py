"""
ask.py
Ask questions about Supabase docs from the terminal.
Retrieves relevant chunks from ChromaDB, sends them to Groq (Llama 3.3),
and prints an answer with source citations.

Usage:
    python src/ask.py
    python src/ask.py "How do I set up authentication?"
"""
import os
import sys
from pathlib import Path

import chromadb
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
# This is the prompt that tells the LLM HOW to answer.
# It's the single most important thing you can tune for answer quality.
QA_PROMPT = PromptTemplate(
    """\
You are Docsy, a helpful AI assistant that answers questions about Supabase \
using ONLY the documentation excerpts provided below.

RULES:
1. Answer based ONLY on the provided context. If the context doesn't contain \
enough information, say "I don't have enough information in the docs to answer \
that fully" and share what you DO know.
2. After your answer, add a "Sources:" section listing the URLs you used.
3. Be concise but thorough. Use code examples from the docs when relevant.
4. If the user greets you or asks what you can do, introduce yourself briefly.

Context from Supabase documentation:
-----
{context_str}
-----

User question: {query_str}

Answer:"""
)
# ---------------------------------------------------------------------------


def setup() -> VectorStoreIndex:
    """Load the existing ChromaDB index and configure models."""

    # Check API key
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise SystemExit(
            "GROQ_API_KEY not found. Make sure your .env file exists and has the key."
        )

    # Check index exists
    if not CHROMA_DIR.exists():
        raise SystemExit(
            "ChromaDB not found. Run 'python src/build_index.py' first."
        )

    # Embedding model (same one used during indexing — MUST match)
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    # LLM via Groq (free, fast)
    Settings.llm = Groq(
        model="llama-3.3-70b-versatile",
        api_key=api_key,
        temperature=0.1,  # low = more factual, less creative
    )

    # Load existing ChromaDB
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    chroma_collection = chroma_client.get_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    index = VectorStoreIndex.from_vector_store(vector_store)

    print(f"✅ Loaded index with {chroma_collection.count()} chunks.\n")
    return index


def ask(index: VectorStoreIndex, question: str) -> str:
    """Ask a question and return the answer with sources."""
    query_engine = index.as_query_engine(
        similarity_top_k=5,           # retrieve top 5 most relevant chunks
        text_qa_template=QA_PROMPT,
    )

    response = query_engine.query(question)

    # Extract source URLs from the retrieved nodes
    sources = set()
    for node in response.source_nodes:
        url = node.metadata.get("source_url", "")
        if url:
            sources.add(url)

    # Build formatted output
    answer = str(response).strip()
    if sources:
        answer += "\n\n📚 Sources:"
        for url in sorted(sources):
            answer += f"\n   → {url}"

    return answer


def interactive_mode(index: VectorStoreIndex):
    """Chat loop for the terminal."""
    print("=" * 50)
    print("💬 Docsy — Ask anything about Supabase docs")
    print("   Type 'quit' or 'exit' to stop.")
    print("=" * 50)

    while True:
        try:
            question = input("\n🔍 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye! 👋")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Bye! 👋")
            break

        print("\n⏳ Thinking...\n")
        answer = ask(index, question)
        print(f"🤖 Docsy: {answer}")


if __name__ == "__main__":
    idx = setup()

    # If a question was passed as a command line argument, answer it and exit
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        print(f"🔍 You: {question}\n")
        print(f"🤖 Docsy: {ask(idx, question)}")
    else:
        # Otherwise, start interactive chat
        interactive_mode(idx)
