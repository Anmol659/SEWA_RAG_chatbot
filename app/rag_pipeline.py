import os
import time
import requests
import json
from threading import Thread
from queue import Queue

from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# --- Configuration ---
VECTORSTORE_PATH = "vectorstore/faiss_index_agri"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEVICE = "cpu"

# --- Ollama API Config ---
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "gemma:2b"   # ✅ uses your local Ollama Gemma

# --- Streaming Callback ---
class QueueCallbackHandler:
    """Custom handler that pushes streamed tokens into a queue from Ollama."""
    def __init__(self, q):
        self.q = q

def create_rag_chain():
    """Initializes retriever (FAISS + embeddings)."""
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': DEVICE}
    )

    if not os.path.exists(VECTORSTORE_PATH):
        raise FileNotFoundError(
            f"Vector store not found at {VECTORSTORE_PATH}. Please run create_vectorstore.py first."
        )

    vectorstore = FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    return retriever

def build_prompt(context: str, question: str) -> str:
    """Builds the prompt for Gemma with retrieved context."""
    return f"""
You are an expert agricultural extension officer for smallholder farmers in Punjab.

INSTRUCTIONS:
1) Use the provided CONTEXT to answer the user's QUESTION.
    - If the CONTEXT contains direct, actionable information that answers the question, first give a concise DIRECT ANSWER (1–2 sentences).
    - Then put a line with exactly three dashes: ---
    - Then give a detailed, numbered STEP-BY-STEP EXPLANATION that references the context (quote brief snippets or say which document supports each step).
    - End with a short "Sources" section listing the context documents used.

2) If the CONTEXT is empty or does not contain actionable information for this QUESTION:
    - Still provide a concise DIRECT ANSWER (1–2 sentences) based on widely accepted agronomic best practices suitable for Punjab.
    - Then put a line with exactly three dashes: ---
    - Provide a numbered, practical STEP-BY-STEP GUIDANCE section that:
      • Recommends what to do immediately (e.g., get a soil test, take a sample).
      • Gives conservative, non-absolute guidance (e.g., mention types of fertilizer to consider, typical application methods like split N application, basal P application), and **avoid absolute numeric mandates**. If giving numbers, use clear ranges and units and label them as "typical range — confirm with soil test".
      • Explicitly states safety/legal cautions and when to consult local extension services.

3) ALWAYS:
    - Answer in the same language as the user's question.
    - Avoid over-confident absolute statements; use hedging when uncertain.
    - Keep the DIRECT ANSWER short and actionable.

Context:
{context}

Question: {question}

Answer:
"""

def stream_rag_response(query: str):
    """Generator that streams response from Ollama using FAISS + Gemma."""
    q = Queue()

    def run_chain_in_thread():
        try:
            retriever = create_rag_chain()
            docs = retriever.get_relevant_documents(query)
            context = "\n".join([doc.page_content for doc in docs])

            prompt = build_prompt(context, query)

            with requests.post(
                OLLAMA_URL,
                json={"model": MODEL, "prompt": prompt, "stream": True},
                stream=True,
            ) as r:
                for line in r.iter_lines():
                    if line:
                        try:
                            data = line.decode("utf-8")
                            # Check if the line is a valid JSON object
                            if data.startswith("{") and data.endswith("}"):
                                obj = json.loads(data)
                                if "response" in obj:
                                    q.put(obj["response"])
                        except (json.JSONDecodeError, UnicodeDecodeError):
                            # Ignore lines that are not valid JSON or can't be decoded
                            continue
            q.put(None)
        except Exception as e:
            print(f"ERROR in RAG chain thread: {e}")
            q.put(e)

    thread = Thread(target=run_chain_in_thread, daemon=True)
    thread.start()

    while True:
        token = q.get()
        if token is None:
            break
        if isinstance(token, Exception):
            raise token
        yield token
        time.sleep(0.01)

    thread.join()
