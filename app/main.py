from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import os
from typing import Optional
from pydantic import BaseModel

# Assume these are your existing modules for the RAG pipeline and tools.
# No changes are needed in these files.
from app.rag_pipeline import stream_rag_response
from tools.market_price_tool import get_mandi_prices
from tools.soil_weather_tool import get_soil_data, PUNJAB_COORDINATES

# --- Application Setup ---
app = FastAPI(
    title="Smart Crop Advisory API",
    description="A multilingual RAG chatbot for farmers that accepts text queries.",
    version="2.1.1" # Version updated for CORS support
)

# --- Add CORS Middleware ---
# This is necessary to allow front-end web applications to communicate with this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8082",                 # React local dev
        "https://e0081e71da4a.ngrok-free.app"    # current ngrok tunnel
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request Body Model ---
class QueryRequest(BaseModel):
    query: str

# --- Keywords for Tool Routing ---
WEATHER_KEYWORDS = ["weather", "temperature", "climate", "forecast", "mausam", "tapman", "jalvayu", "ਮੌਸਮ", "ਤਾਪਮਾਨ"]
PRICE_KEYWORDS = ["price", "rate", "mandi", "bhav", "kimat", "bazaar", "ਮੁੱਲ", "ਕੀਮਤ", "ਮੰਡੀ", "ਭਾਅ"]
KNOWN_LOCATIONS = list(PUNJAB_COORDINATES.keys())
KNOWN_COMMODITIES = ["wheat", "rice", "paddy", "cotton", "maize", "ਕਣਕ", "ਝੋਨਾ", "ਨਰਮਾ", "ਮੱਕੀ", "गेहूं", "धान", "कपास", "मक्का"]

def extract_entity(query: str, entity_list: list) -> Optional[str]:
    """
    Extracts the first matching known entity (like a commodity or location) from a query.
    Also handles translation of common terms to a standardized English keyword.
    """
    for entity in entity_list:
        if entity in query:
            # Standardize commodity names
            if entity in ["ਝੋਨਾ", "धान"]: return "rice"
            if entity in ["ਕਣਕ", "गेहूं"]: return "wheat"
            if entity in ["ਨਰਮਾ", "कपास"]: return "cotton"
            if entity in ["ਮੱਕੀ", "मक्का"]: return "maize"
            return entity
    return None

@app.post("/ask")
async def ask_endpoint(request: QueryRequest):
    """
    This endpoint processes a text-based query from the user, expecting a JSON body
    with a "query" field (e.g., {"query": "How to manage wheat rust?"}).

    The final response will be generated in the same language as the input query.
    """
    final_query_text = request.query.lower()
    print(f"INFO: Processing query: '{final_query_text}'")

    # Tool Routing: Weather/Soil Data
    if any(k in final_query_text for k in WEATHER_KEYWORDS):
        location = extract_entity(final_query_text, KNOWN_LOCATIONS)
        if location:
            report = get_soil_data(location)
            async def gen_report(): yield report
            return StreamingResponse(gen_report(), media_type="text/plain")
        # If keyword found but no location, fallback to RAG

    # Tool Routing: Market Prices
    if any(k in final_query_text for k in PRICE_KEYWORDS):
        commodity = extract_entity(final_query_text, KNOWN_COMMODITIES)
        if commodity:
            report = get_mandi_prices("punjab", commodity)
            async def gen_report(): yield report
            return StreamingResponse(gen_report(), media_type="text/plain")
        # If keyword found but no commodity, fallback to RAG

    # --- Default to RAG Pipeline ---
    # If no specific tools are triggered, use the RAG pipeline for a general response.
    print("INFO: No specific tool triggered. Routing to RAG pipeline.")
    return StreamingResponse(stream_rag_response(final_query_text), media_type="text/event-stream")

@app.on_event("startup")
async def on_startup():
    print("===== Application Startup =====")
    print("INFO: FastAPI server with text-only /ask endpoint is ready.")

