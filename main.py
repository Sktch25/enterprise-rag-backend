import os
import shutil
import subprocess
import sys
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from query_db import run_rag_pipeline  

app = FastAPI(
    title="Enterprise RAG Backend",
    description="A production-ready API for querying document vector databases",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    question: str
    answer: str

@app.get("/")
def read_root():
    return {"status": "online", "message": "Enterprise RAG API is running smoothly."}

@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Accepts a user-uploaded PDF and saves it to the local data directory
    so it can be ingested into the vector database.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    os.makedirs("data", exist_ok=True)
    
    save_path = os.path.join("data", file.filename)
    
    try:
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        subprocess.run([sys.executable, "build_db.py"], check=True)        
        return {"filename": file.filename, "message": "File uploaded and vector database successfully updated!"}
        
    except subprocess.CalledProcessError:
        raise HTTPException(status_code=500, detail="File saved, but vector database failed to build.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not save file: {str(e)}")
    
@app.post("/api/query")
async def handle_query(request: QueryRequest):
    """
    Accepts a question from the frontend, searches the ChromaDB 
    vector store, and returns the AI-generated answer.
    """
    try:
        answer = run_rag_pipeline(request.question)
        
        return {"question": request.question, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process query: {str(e)}")