from langchain.vectorstores import Chroma
from langchain.embeddings import  GoogleGenerativeAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
CHROMA_DIR='resume_vector_store'
def get_vector_store():
    embeddings = GoogleGenerativeAIEmbeddings()
    return Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
def add_resume_to_vectorstore(resume_path):
    loader = TextLoader(resume_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)
    vectordb = get_vector_store()
    vectordb.add_documents(chunks)
    vectordb.persist()
