import os
import time
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

def build_vector_database():
    print("Scanning data/ folder for PDFs...")
    
    loader = PyPDFDirectoryLoader("data")
    documents = loader.load()
    
    if not documents:
        print("No PDFs found in the data/ folder.")
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} total text chunks.")

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")    
    db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

    batch_size = 90  
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        print(f"Uploading chunks {i} to {i + len(batch)} into ChromaDB...")
        db.add_documents(batch)
        
        if i + batch_size < len(chunks):
            print("Sleeping for 60 seconds to respect Gemini API rate limits...")
            time.sleep(60)
            
    print("Vector database successfully built and updated!")

if __name__ == "__main__":
    build_vector_database()