import os
import time
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- Configuration ---
DATA_PATH = "data/"
VECTORSTORE_PATH = "vectorstore/faiss_index_agri"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

def create_vector_store():
    """
    Loads documents from the data directory, splits them into chunks,
    creates embeddings, and saves them to a FAISS vector store.
    """
    print("--- Starting Vector Store Creation Process ---")

    if not os.path.exists(DATA_PATH):
        print(f"Error: Data directory not found at '{DATA_PATH}'")
        return

    # 1. Load Documents
    print(f"1. Loading documents from '{DATA_PATH}'...")
    start_time = time.time()
    loader = DirectoryLoader(
        DATA_PATH,
        glob="**/*.pdf",   # Recursively search for PDFs
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    if not documents:
        print("No PDF documents found. Please check the data directory.")
        return
    end_time = time.time()
    print(f"   Loaded {len(documents)} documents in {end_time - start_time:.2f} seconds.")

    # 2. Split Documents into Chunks
    print("\n2. Splitting documents into manageable chunks...")
    start_time = time.time()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)
    end_time = time.time()
    print(f"   Split into {len(docs)} chunks in {end_time - start_time:.2f} seconds.")

    # 3. Create Embeddings
    print(f"\n3. Creating embeddings using '{EMBEDDING_MODEL}'...")
    print("   (This may take a few minutes, depending on the number of documents and your hardware)...")
    start_time = time.time()
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'}  # Force CPU for broad compatibility
    )
    end_time = time.time()
    print(f"   Embedding model loaded in {end_time - start_time:.2f} seconds.")

    # 4. Create and Save FAISS Vector Store
    print("\n4. Creating FAISS vector store from document chunks...")
    start_time = time.time()
    vectorstore = FAISS.from_documents(docs, embeddings)

    os.makedirs(os.path.dirname(VECTORSTORE_PATH), exist_ok=True)
    vectorstore.save_local(VECTORSTORE_PATH)
    end_time = time.time()
    print(f"   Vector store created and saved to '{VECTORSTORE_PATH}' in {end_time - start_time:.2f} seconds.")

    print("\n--- Vector Store Creation Complete! ---")

if __name__ == "__main__":
    create_vector_store()
