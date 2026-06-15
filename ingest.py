import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter

def ingest_and_chunk_pdf(pdf_path: str):
    print(f"Opening document: {pdf_path}")
    doc = fitz.open(pdf_path)
    
    full_text = ""
    
    # 1. Extract all text into a single string
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text("text")
        if text:
            full_text += text + "\n"
            
    print(f"Extraction complete. Total characters read: {len(full_text)}")
    
    # 2. Define our Chunking Strategy
    # chunk_size: Max characters per chunk
    # chunk_overlap: Characters shared between consecutive chunks to keep context
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len,
        is_separator_regex=False,
    )
    
    # 3. Split the text
    chunks = text_splitter.split_text(full_text)
    print(f"Chunking complete. Created {len(chunks)} total chunks.")
    
    # 4. Inspect the first 3 chunks to verify quality
    print("\n" + "="*20 + " INSPECTING FIRST 3 CHUNKS " + "="*20)
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n[CHUNK {i + 1}] (Length: {len(chunk)} characters):")
        print("-" * 40)
        print(chunk)
        print("-" * 40)

if __name__ == "__main__":
    SAMPLE_PDF = "data/sample_document.pdf"
    ingest_and_chunk_pdf(SAMPLE_PDF)