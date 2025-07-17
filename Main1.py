import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import pipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

#steps 1: Load the PDF document
loader = PyPDFLoader("Data/INVOICE.pdf")
documents = loader.load()

#step 2: split the documents into smaller chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10)

chunks = splitter.split_documents(documents)

#step 3: Use Hugging Face
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

#step 4: Embed and Store in FAISS
vectorstore = FAISS.from_documents(chunks, embedding_model)
vectorstore.save_local("vectorstore")

#step 5: load retriever and ask questions 
retriever = vectorstore.as_retriever()
# Load Hugging Face QA model
qa_model = pipeline("text2text-generation", model="google/flan-t5-base")


while True: 
    query = input("Ask a question: (or type 'exit'): ")
    if query.lower() == 'exit':
        break
        
     # Retrieve relevant chunks
    docs = retriever.get_relevant_documents(query)
    context = " ".join([doc.page_content for doc in docs])

    # Format prompt for FLAN-T5
    prompt = f"Answer the question based on the context:\nContext: {context}\nQuestion: {query}"

    # Get answer from HF model
    result = qa_model(prompt, max_length=256, do_sample=False)[0]["generated_text"]

    print("\n📚 Answer:", result)

