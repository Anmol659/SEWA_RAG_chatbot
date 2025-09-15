import os
import shutil
import re
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import Optional, List

# Import schemas and the RAG chain creation function
from .schemas import QueryRequest
from .rag_pipeline import create_rag_chain

# Import tool functions from the 'tools' directory
from tools.soil_weather_tool import get_soil_data, PUNJAB_COORDINATES
from tools.market_price_tool import get_mandi_prices
from tools.pest_detection_tool import get_pest_prediction

# --- Application Setup ---
app = FastAPI(
    title="Smart Crop Advisory API",
    description="An intelligent, multimodal chatbot backend for farmers in Punjab.",
    version="1.2.0"
)

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global RAG Chain Variable ---
rag_chain = None

@app.on_event("startup")
def startup_event():
    """Load the RAG chain model on application startup."""
    global rag_chain
    print("--- Loading RAG Chain on startup ---")
    try:
        rag_chain = create_rag_chain()
        print("--- RAG Chain loaded successfully ---")
    except Exception as e:
        print(f"FATAL: Could not load RAG chain on startup. Error: {e}")
        rag_chain = None

# --- Helper functions for Intent Detection ---
KNOWN_COMMODITIES = ["wheat", "rice", "paddy", "cotton", "maize", "sugarcane"]

def find_keywords(query: str, keywords: List[str]) -> Optional[str]:
    """Finds the first matching keyword from a list in the query."""
    for keyword in keywords:
        if re.search(r'\b' + re.escape(keyword) + r'\b', query):
            return keyword
    return None

# --- API Endpoints ---

@app.get("/", summary="Root endpoint for health check")
def read_root():
    """A simple health check endpoint to confirm the API is running."""
    return {"status": "Smart Crop Advisory API is running."}

@app.post("/ask", summary="Unified endpoint for all queries (Text, Voice, Tools)")
async def ask_question(request: QueryRequest):
    """
    This intelligent endpoint determines the user's intent and routes the query
    to the appropriate tool (live data) or the RAG pipeline (knowledge base).
    """
    query_lower = request.query.lower()

    # --- Intent 1: Check for Weather or Soil Data Request ---
    if any(keyword in query_lower for keyword in ["weather", "temperature", "soil", "moisture", "climate today", "मौसम", "तापमान", "मिट्टी", "नमी"]):
        location = find_keywords(query_lower, list(PUNJAB_COORDINATES.keys()))
        if location:
            print(f"INFO: Weather/soil intent detected for location: {location}")
            live_data_report = get_soil_data(location)
            return {"answer": live_data_report, "source": "Live Weather API"}

    # --- Intent 2: Check for Market Price Request ---
    if any(keyword in query_lower for keyword in ["price", "mandi", "rate", "bhav", "कीमत", "मंडी", "भाव"]):
        commodity = find_keywords(query_lower, KNOWN_COMMODITIES)
        if commodity:
            print(f"INFO: Market price intent detected for commodity: {commodity}")
            # Assuming Punjab for this project context
            price_report = get_mandi_prices("punjab", commodity)
            return {"answer": price_report, "source": "Live Market Price API"}

    # --- Default: Use RAG for all other knowledge-based questions ---
    print("INFO: No specific tool intent detected. Routing to RAG pipeline.")
    if rag_chain is None:
        raise HTTPException(status_code=503, detail="RAG model is not available. Please check server logs.")
    try:
        result = rag_chain.invoke({"query": request.query})
        return {"answer": result.get('result', 'No result found.'), "source": "Document Knowledge Base"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during RAG processing: {e}")


@app.post("/predict-pest", summary="Identify crop disease from an image")
async def predict_pest(file: UploadFile = File(...)):
    """Handles image uploads for pest and disease detection."""
    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    temp_file_path = os.path.join(temp_dir, file.filename)

    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        prediction_result = get_pest_prediction(temp_file_path)
        return {"prediction": prediction_result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during pest prediction: {e}")
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

# Keeping these direct tool endpoints can be useful for specific app features or debugging
@app.get("/market-prices", summary="Directly get latest mandi prices")
def market_prices(state: str, commodity: str):
    """Directly calls the market price scraping tool."""
    try:
        price_report = get_mandi_prices(state, commodity)
        return {"report": price_report}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/soil-data", summary="Directly get recent soil data")
def soil_data(location: str):
    """Directly calls the soil data tool."""
    try:
        soil_report = get_soil_data(location)
        return {"report": soil_report}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("To run the API server, use the command from your project root:")
    print("uvicorn app.main:app --reload --host 0.0.0.0")
    uvicorn.run(app, host="0.0.0.0", port=8000)

