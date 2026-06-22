import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

def run_rag_pipeline(question: str):
    db_path = "./chroma_db"
    
    if not os.path.exists(db_path):
        print(f"Error: Database folder '{db_path}' not found. Run build_db.py first.")
        return

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")    
    vector_db = Chroma(persist_directory=db_path, embedding_function=embeddings)
    
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0)    
    template = """You are an advanced enterprise AI assistant. Answer the user's question using ONLY the provided context below. If the context does not contain the answer, say "I cannot find the answer in the provided document." Do not make things up.

Context:
{context}

Question: {question}

Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    print(f"\n🔍 Question: {question}")
    print("Searching database and generating answer...")
    
    response = rag_chain.invoke(question)
    return response
if __name__ == "__main__":
    print("--- Enterprise RAG System ---")
    
    while True:
        user_question = input("Ask a question about your document (or type 'exit' to quit): ")
        
        if user_question.lower() == 'exit':
            print("Exiting RAG system. Goodbye!")
            break
            
        if not user_question.strip():
            continue
            
        run_rag_pipeline(user_question)