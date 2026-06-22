import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from query_db import run_rag_pipeline  # Importing your existing core function

app = FastAPI(
    title="Enterprise RAG Backend",
    description="A production-ready API for querying document vector databases",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (POST, GET, etc.)
    allow_headers=["*"],  # Allows all headers
)
# Define the expected JSON structure for incoming requests
class QueryRequest(BaseModel):
    question: str

# Define the structured JSON response layout
class QueryResponse(BaseModel):
    question: str
    answer: str

@app.get("/")
def read_root():
    return {"status": "online", "message": "Enterprise RAG API is running smoothly."}

@app.post("/api/query", response_model=QueryResponse)
def query_endpoint(payload: QueryRequest):
    """
    Accepts a user question, runs it through the vector database 
    and Gemini LLM, and returns the strictly grounded answer.
    """
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    
    try:
        # Execute your working pipeline logic
        # Note: If your run_rag_pipeline prints instead of returning, 
        # make sure it returns the string response at the end of its block.
        answer = run_rag_pipeline(payload.question)
        
        return QueryResponse(
            question=payload.question,
            answer=answer
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")