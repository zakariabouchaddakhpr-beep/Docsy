"""
build_index.py
Reads all scraped .txt files, chunks them, generates embeddings,
and stores everything in a local ChromaDB vector database.

This only needs to run ONCE (or again if you re-scrape new docs).
Takes ~2-5 minutes depending on your CPU (embeddings run locally).

Usage:
    python src/build_index.py
"""
import json
from pathlib import Path

import chromadb
from llama_index.core import (
    Document,
    Settings,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

# ---- Paths ----------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
PAGES_DIR = BASE_DIR / "data" / "pages"
METADATA_FILE = BASE_DIR / "data" / "metadata.json"
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "docsy_docs"
# ---------------------------------------------------------------------------


def load_documents() -> list[Document]:
    """Load .txt files and attach metadata (source URL + title)."""
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    documents = []
    loaded = 0

    for txt_file in sorted(PAGES_DIR.glob("*.txt")):
        text = txt_file.read_text(encoding="utf-8").strip()
        if not text:
            continue

        file_info = metadata.get(txt_file.name, {})
        source_url = file_info.get("url", "unknown")
        title = file_info.get("title", txt_file.stem)

        doc = Document(
            text=text,
            metadata={
                "source_url": source_url,
                "title": title,
                "filename": txt_file.name,
            },
        )
        documents.append(doc)
        loaded += 1

    print(f"📄 Loaded {loaded} documents from {PAGES_DIR}")
    return documents


def build_index(documents: list[Document]):
    """Chunk documents, embed them, store in ChromaDB."""

    # 1. Configure the embedding model (runs locally, no API needed)
    print("🧠 Loading embedding model (first run downloads ~130MB)...")
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.embed_model = embed_model

    # We don't need an LLM for indexing, just embeddings
    Settings.llm = None

    # 2. Set up the chunker
    #    - chunk_size=512: each chunk is ~512 tokens (~400 words)
    #    - chunk_overlap=50: chunks share 50 tokens at boundaries
    #      so context isn't lost at chunk edges
    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)

    # 3. Set up ChromaDB (local, persisted to disk)
    print("💾 Setting up ChromaDB...")
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Delete existing collection if re-running (clean slate)
    try:
        chroma_client.delete_collection(COLLECTION_NAME)
        print("   (Deleted old collection)")
    except Exception:
        pass

    chroma_collection = chroma_client.create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 4. Build the index (this does the chunking + embedding)
    print(f"⚙️  Chunking and embedding {len(documents)} documents...")
    print("   This takes 2-5 minutes on CPU. Grab a coffee ☕")

    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        transformations=[splitter],
        show_progress=True,
    )

    print("\n" + "=" * 50)
    print(f"✅ Index built! Stored in {CHROMA_DIR}")
    print(f"   Collection: {COLLECTION_NAME}")
    print(f"   Total chunks: {chroma_collection.count()}")
    print("=" * 50)
    print("\nNext step: python src/ask.py")

    return index


if __name__ == "__main__":
    docs = load_documents()
    build_index(docs)
