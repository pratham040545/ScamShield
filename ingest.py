import os
import shutil
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import chromadb

load_dotenv()

DATA_DIR = "./data"
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "scamshield_kb"
CHUNK_SIZE = 600
CHUNK_OVERLAP = 80

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def load_documents():
    documents = []
    for filename in sorted(os.listdir(DATA_DIR)):
        if filename.endswith(".txt"):
            filepath = os.path.join(DATA_DIR, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            doc = Document(
                page_content=content,
                metadata={"source": filename}
            )
            documents.append(doc)
            print(f"  ✅ Loaded: {filename} ({len(content):,} chars)")
    return documents


def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n===", "\n\n", "\n", ". ", " "]
    )
    chunks = splitter.split_documents(documents)
    print(f"  ✅ Total chunks: {len(chunks)}")
    return chunks


def ingest():
    print("\n🛡️  ScamShield — Knowledge Base Ingestion")
    print("=" * 45)

    print("\n📂 Loading documents...")
    documents = load_documents()

    print("\n✂️  Chunking documents...")
    chunks = chunk_documents(documents)

    print("\n🔢 Creating embeddings...")
    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]
    embeddings = embedding_model.encode(texts, show_progress_bar=True).tolist()
    print(f"  ✅ Embeddings created: {len(embeddings)}")

    print("\n💾 Storing in ChromaDB...")
    if os.path.exists(CHROMA_DIR):
        print("  🗑️  Clearing old ChromaDB...")
        shutil.rmtree(CHROMA_DIR)

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.create_collection(name=COLLECTION_NAME)
    ids = [str(i) for i in range(len(texts))]
    collection.add(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=embeddings
    )
    print(f"  ✅ Stored {collection.count()} vectors in ChromaDB")
    print("\n🚀 Done! Run: streamlit run app.py\n")


if __name__ == "__main__":
    ingest()
