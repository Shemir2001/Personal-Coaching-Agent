from langchain_community.vectorstores import Chroma

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv
load_dotenv()
CHROMA_DIR='resume_vector_store'
def get_vector_store():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.environ["API_KEY"])
    return Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
def add_resume_to_vectorstore(resume_path):
    loader = UnstructuredFileLoader(resume_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)
    vectordb = get_vector_store()
    vectordb.add_documents(chunks)
    vectordb.persist()
