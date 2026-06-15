import os
import time
import fitz  # PyMuPDF
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

# 1. Load the API key
load_dotenv()

def build_vector_database(pdf_path: str, db_path: str = "./chroma_db"):
    print(f"Opening document: {pdf_path}")
    doc = fitz.open(pdf_path)
    
    full_text = ""
    for page in doc:
        text = page.get_text("text")
        if text:
            full_text += text + "\n"
            
    print("Chunking text...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )
    chunks = text_splitter.split_text(full_text)
    print(f"Created {len(chunks)} chunks.")
    
    print("Connecting to Gemini Embedding Model...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    
    print("Building Chroma Vector Database in batches to respect API limits...")
    # Initialize an empty database
    vector_db = Chroma(persist_directory=db_path, embedding_function=embeddings)
    
    # 3. Process chunks in small batches with a 5-second pause
    batch_size = 10
    total_batches = (len(chunks) // batch_size) + 1
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        current_batch = (i // batch_size) + 1
        print(f"  -> Uploading batch {current_batch} of {total_batches}...")
        
        vector_db.add_texts(texts=batch)
        
        # Pause for 5 seconds between batches unless it's the very last one
        if current_batch < total_batches:
            time.sleep(5)
            
    print(f"\n✅ Success! Vector database permanently saved to the '{db_path}' folder.")

if __name__ == "__main__":
    SAMPLE_PDF = "data/sample_document.pdf"
    build_vector_database(SAMPLE_PDF)