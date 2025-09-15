import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Import schemas and the RAG chain creation function
from .schemas import QueryRequest
from .rag_pipeline import create_rag_chain

# Import tool functions
from tools.soil_weather_tool import get_soil_data
from tools.market_price_tool import get_mandi_prices
from tools.pest_detection_tool import get_pest_prediction

# --- Application Setup ---
app = FastAPI(
    title="Smart Crop Advisory API",
    description="An AI-powered chatbot backend for farmers in Punjab, providing advice on crops, pests, soil, and market prices.",
    version="1.0.0"
)

# --- CORS Middleware ---
# Allows the frontend (your mobile app) to communicate with this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for simplicity, can be restricted in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global variable for the RAG chain ---
# We load the model and vector store only once when the application starts.
rag_chain = None

@app.on_event("startup")
def startup_event():
    """
    Load the RAG chain model on application startup.
    This is an efficient way to manage resources.
    """
    global rag_chain
    print("--- Loading RAG Chain on startup ---")
    try:
        rag_chain = create_rag_chain()
        print("--- RAG Chain loaded successfully ---")
    except Exception as e:
        print(f"FATAL: Could not load RAG chain on startup. Error: {e}")
        # In a real-world scenario, you might want the app to exit or handle this more gracefully.
        rag_chain = None

# --- API Endpoints ---

@app.get("/", summary="Root endpoint for health check")
def read_root():
    """A simple health check endpoint to confirm the API is running."""
    return {"status": "Smart Crop Advisory API is running."}

@app.post("/ask", summary="Get answers from the RAG chatbot")
def ask_question(request: QueryRequest):
    """
    This is the main endpoint for text-based queries (including speech-to-text input from a mobile app).
    It takes a user's question, processes it through the RAG pipeline, and returns a context-aware answer.
    
    **Voice Input Handling:** Your mobile app should convert the user's voice note to text
    using a Speech-to-Text (STT) service and send the resulting text to this endpoint.
    """
    if rag_chain is None:
        raise HTTPException(status_code=503, detail="RAG model is not available. Please check server logs.")
    try:
        result = rag_chain.invoke({"query": request.query})
        return {"answer": result['result']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while processing the query: {e}")

@app.post("/predict-pest", summary="Identify crop disease from an image")
async def predict_pest(file: UploadFile = File(...)):
    """
    This endpoint handles image uploads for pest and disease detection.
    It saves the uploaded image temporarily, runs the prediction model,
    and returns the identified disease with a confidence score.
    """
    # Create a temporary directory to store uploads if it doesn't exist
    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    
    temp_file_path = os.path.join(temp_dir, file.filename)

    try:
        # Save the uploaded file
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Get prediction from the tool
        prediction_result = get_pest_prediction(temp_file_path)
        
        # The text result (e.g., "Detected Disease: Rice - Leaf blast") can now be used
        # in a follow-up query to the /ask endpoint to get treatment advice.
        
        return {"prediction": prediction_result}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during pest prediction: {e}")
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.get("/market-prices", summary="Get latest mandi prices for a commodity")
def market_prices(state: str, commodity: str):
    """
    Calls the market price scraping tool to get the latest prices.
    Example: /market-prices?state=punjab&commodity=wheat
    """
    try:
        price_report = get_mandi_prices(state, commodity)
        return {"report": price_report}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/soil-data", summary="Get recent soil data for a location")
def soil_data(location: str):
    """
    Calls the soil data tool to get recent soil moisture and temperature.
    Example: /soil-data?location=Ludhiana
    """
    try:
        soil_report = get_soil_data(location)
        return {"report": soil_report}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # This allows you to run the API directly for local testing.
    # Use the command: uvicorn app.main:app --reload
    print("To run the API server, use the command: uvicorn app.main:app --reload")
