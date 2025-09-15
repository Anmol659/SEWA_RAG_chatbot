from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    """
    Defines the structure for incoming API requests for the /ask endpoint.
    It expects a single field 'query' which is a non-empty string.
    """
    query: str = Field(..., min_length=1, description="The user's question to the RAG chatbot.")

class ImagePredictionRequest(BaseModel):
    """
    Defines the structure for the pest prediction endpoint.
    It expects a file path to an image.
    """
    image_path: str = Field(..., description="The local path to the user's uploaded image.")
