import os
import torch
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import CTransformers
from langchain.chains.retrieval_qa.base import RetrievalQA

# --- Configuration ---
VECTORSTORE_PATH = "vectorstore/faiss_index_agri"
MODEL_PATH = "models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# --- Device Setup ---
DEVICE = "cpu"
print(f"🔹 Forcing device: {DEVICE}")

def load_llm():
    """Loads the quantized Mistral-7B model using CTransformers."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"LLM model file not found at {MODEL_PATH}. "
            "Please download it and place it in the 'models' directory."
        )
    
    llm = CTransformers(
        model=MODEL_PATH,
        model_type="mistral",
        config={
            'max_new_tokens': 1024,
            'temperature': 0.7,
            'context_length': 4096
        },
        # Explicitly set gpu_layers to 0 to force CPU usage
        gpu_layers=0  
    )
    return llm

def create_rag_chain():
    """Initializes and returns the RAG components."""
    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': DEVICE}
    )
    print("✅ Embedding model loaded.")
    
    if not os.path.exists(VECTORSTORE_PATH):
        raise FileNotFoundError(
            f"Vector store not found at {VECTORSTORE_PATH}. "
            "Please run create_vectorstore.py first."
        )

    print("Loading vector store...")
    vectorstore = FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    print("✅ Vector store loaded.")
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    prompt_template = """
    First, provide a direct and concise answer to the user's question about farming in Punjab using the context below.
    Then, add the separator '---' and provide a more detailed, step-by-step explanation based on the same context.
    If you don't find the answer in the context, state that the information is not in your knowledge base.
    Always answer in the same language as the user's question.

    Context: {context}
    Question: {question}

    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    print("Loading LLM... (This may take a moment)")
    llm = load_llm()
    print("✅ LLM loaded.")

    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    
    return rag_chain

def stream_rag_response(query: str, chain: RetrievalQA):
    """
    Performs the RAG steps and yields the LLM's response token by token.
    
    Args:
        query: The user's question.
        chain: The pre-configured RetrievalQA chain.

    Yields:
        str: Each token of the generated response.
    """
    try:
        # 1. Retrieve relevant documents
        docs = chain.retriever.invoke(query)
        
        if not docs:
            yield "I couldn't find specific information on that topic in my knowledge base."
            return

        # 2. Format the context for the prompt
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # 3. Create the final prompt with the retrieved context
        final_prompt = chain.combine_documents_chain.llm_chain.prompt.format(context=context, question=query)

        # 4. Stream the response from the LLM
        llm = chain.combine_documents_chain.llm_chain.llm
        for token in llm.stream(final_prompt):
            yield token
    except Exception as e:
        print(f"Error during streaming: {e}")
        yield "An error occurred while generating the response."

# This block is for direct testing and will not be used by the main FastAPI app.
if __name__ == "__main__":
    print("--- Testing RAG Pipeline Streaming ---")
    try:
        chain = create_rag_chain()
        print("\n✅ RAG chain created successfully.")
        
        query_en = "What is the best fertilizer for wheat in Punjab?"
        print(f"\nTesting with query: '{query_en}'")
        
        print("\n--- Streamed Response ---")
        response_stream = stream_rag_response(query_en, chain)
        for token in response_stream:
            print(token, end="", flush=True)
        print("\n--- End of Stream ---")
        
    except Exception as e:
        print(f"\n❌ An error occurred during testing: {e}")

