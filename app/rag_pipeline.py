from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import CTransformers
from langchain.chains.retrieval_qa.base import RetrievalQA
import os

# --- Configuration ---
VECTORSTORE_PATH = "vectorstore/faiss_index_agri"
MODEL_PATH = "models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

def load_llm():
    """
    Loads the quantized Mistral-7B model using CTransformers.
    Configured for CPU execution and optimized for a balance of speed and quality.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"LLM model file not found at {MODEL_PATH}. Please download it from Hugging Face.")
        
    llm = CTransformers(
        model=MODEL_PATH,
        model_type="mistral",
        config={'max_new_tokens': 1024, 'temperature': 0.7, 'context_length': 4096}
    )
    return llm

def create_rag_chain():
    """
    Initializes and returns the complete RAG (Retrieval-Augmented Generation) chain.
    This function sets up the retriever, the prompt template, and combines them
    with the LLM to form the question-answering system.
    """
    # 1. Load Embeddings and Vector Store
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'}
    )
    
    if not os.path.exists(VECTORSTORE_PATH):
         raise FileNotFoundError(f"Vector store not found at {VECTORSTORE_PATH}. Please run create_vectorstore.py first.")

    vectorstore = FAISS.load_local(VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True)
    
    # 2. Create Retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 3. Define the Prompt Template (Modified for two-part answers)
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

    # 4. Load the LLM
    llm = load_llm()

    # 5. Create the RAG Chain
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    
    return rag_chain

if __name__ == "__main__":
    # This block is for testing the RAG pipeline directly
    print("--- Testing RAG Pipeline ---")
    try:
        chain = create_rag_chain()
        print("RAG chain created successfully.")
        
        # Example query in English
        query_en = "What is the best fertilizer for wheat in Punjab?"
        print(f"\nTesting with English query: '{query_en}'")
        result_en = chain.invoke({"query": query_en})
        print("\n--- English Response ---")
        print(result_en['result'])

        # Example query in Punjabi
        query_pu = "ਪੰਜਾਬ ਵਿੱਚ ਕਣਕ ਲਈ ਸਭ ਤੋਂ ਵਧੀਆ ਖਾਦ ਕਿਹੜੀ ਹੈ?"
        print(f"\nTesting with Punjabi query: '{query_pu}'")
        result_pu = chain.invoke({"query": query_pu})
        print("\n--- Punjabi Response ---")
        print(result_pu['result'])
        
    except Exception as e:
        print(f"An error occurred during testing: {e}")

