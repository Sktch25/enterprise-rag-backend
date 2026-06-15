import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Load environment variables (API Key)
load_dotenv()

def run_rag_pipeline(question: str):
    db_path = "./chroma_db"
    
    if not os.path.exists(db_path):
        print(f"❌ Error: Database folder '{db_path}' not found. Run build_db.py first.")
        return

    # 2. Reconnect to the exact same embedding model used to build the DB
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    
    # 3. Load the existing database from your hard drive
    vector_db = Chroma(persist_directory=db_path, embedding_function=embeddings)
    
    # 4. Turn the database into a 'Retriever' (configured to grab the top 3 closest chunks)
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    
    # 5. Initialize the language model for answering the question
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0)    
    # 6. Design the strict system prompt template
    template = """You are an advanced enterprise AI assistant. Answer the user's question using ONLY the provided context below. If the context does not contain the answer, say "I cannot find the answer in the provided document." Do not make things up.

Context:
{context}

Question: {question}

Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # 7. Helper function to format retrieved documents nicely for the prompt
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    # 8. Construct the LangChain Expression Language (LCEL) chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    print(f"\n🔍 Question: {question}")
    print("Searching database and generating answer...")
    
    # 9. Execute the chain
    response = rag_chain.invoke(question)
    return response
if __name__ == "__main__":
    print("--- Enterprise RAG System ---")
    
    # This allows an interactive loop so you can ask multiple questions
    while True:
        user_question = input("Ask a question about your document (or type 'exit' to quit): ")
        
        if user_question.lower() == 'exit':
            print("Exiting RAG system. Goodbye!")
            break
            
        if not user_question.strip():
            continue
            
        run_rag_pipeline(user_question)