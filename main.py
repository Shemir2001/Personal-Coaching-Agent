import streamlit as st
from agent_logic import run_agent
from vector_store import add_resume_to_vectorstore

st.set_page_config(page_title="Career Coach AI", page_icon="💼")
st.title("💼 Career Coach AI")

# --- Initialize session state ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # List of (user_input, ai_response)
if "resume_uploaded" not in st.session_state:
    st.session_state.resume_uploaded = False

# --- Sidebar: Resume Upload + Download ---
with st.sidebar:
    st.markdown("## 🤖 Personal Chatbot")
    st.header("Upload your Resume")
    uploaded_file = st.file_uploader("PDF/TXT Resume", type=["pdf", "txt"])
    if uploaded_file:
        with open("resume.txt", "wb") as f:
            f.write(uploaded_file.getbuffer())
        add_resume_to_vectorstore("resume.txt")
        st.session_state.resume_uploaded = True
        st.success("✅ Resume added to memory!")

    # Show download button only if resume was uploaded
    if st.session_state.resume_uploaded:
        with open("resume.txt", "rb") as f:
            resume_bytes = f.read()
        st.download_button(
            label="⬇️ Download Uploaded Resume",
            data=resume_bytes,
            file_name="resume.txt",
            mime="text/plain",
            help="Download your uploaded resume",
            key="download_resume_once"  # Unique key
        )

# --- Chat Input ---
user_input = st.chat_input("Ask me anything about your career, resume, or jobs")

if user_input:
    with st.spinner("🤖 Thinking..."):
        response = run_agent(user_input)

    # Save to session state
    st.session_state.chat_history.append((user_input, response))

# --- Chat Display (like ChatGPT) ---
for user_msg, ai_msg in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(user_msg)
    with st.chat_message("assistant"):
        st.markdown(ai_msg, unsafe_allow_html=True)
    st.markdown("---")  # Separator between messages