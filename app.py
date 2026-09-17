from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


loader=PyPDFLoader("data/iso27001.pdf")
documents=loader.load()

print(f"total pages loaded:{len(documents)}")
print("---Pehle page ka preview---")
print(documents[0].page_content[:300])
print("\n---Metadata---")
print(documents[0].metadata)


text_splitter=RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""]
)

chunks=text_splitter.split_documents(documents)
print(f"\nTotal chunks banaye: {len(chunks)}")
print("--- Pehle chunk ka preview ---")
print(chunks[0].page_content)


#Embedding
print("\nEmbedding model load ho rha hai...")
embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

#vector store 
print("creating vector store...")
vectorstore=Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

print("vector store is ready!")

# Retriever creation
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}   # top 4 relevant chunks 
)

# Step 6: LLM setup
llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)

# Prompt template
prompt = ChatPromptTemplate.from_template("""
Diye gaye context ke basis par question ka jawab do.
Agar context mein answer nahi hai, to bolo "Mujhe iska pata nahi context se".

Context:
{context}

question: {question}
""")
#helper function
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
    #RAG chain (LCEL syntax)
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Step 10: Test 
print("\n--- RAG Bot Ready! ---")
question = "summary of the pdf in 3 lines"  
answer = rag_chain.invoke(question)
print(f"\nSawaal: {question}")
print(f"Jawab: {answer}")