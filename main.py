import streamlit as st
from agent_logic import run_agent
from vector_store import add_resume_to_vectorstore
import os
from dotenv import load_dotenv
load_dotenv()

st.title("💼 Career Coach AI")

with st.sidebar:
    st.header("Upload your Resume")
    uploaded_file = st.file_uploader("PDF/TXT Resume")
    if uploaded_file:
        with open("resume.txt", "wb") as f:
            f.write(uploaded_file.getbuffer())
        add_resume_to_vectorstore("resume.txt")
        st.success("Resume added to memory")

prompt = st.text_input("Ask me anything about your career, resume, or jobs")

if prompt:
    with st.spinner("Thinking..."):
        response = run_agent(prompt)
    st.write(response)
