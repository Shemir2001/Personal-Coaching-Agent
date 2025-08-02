from langchain.agents import initialize_agent, Tool
from langchain_google_genai import ChatGoogleGenerativeAI
from tools import resume_feedback, job_matcher, mock_interview_response
from memory import get_memory
from vector_store import get_vector_store
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    google_api_key=os.environ["API_KEY"]
)

memory = get_memory()
retriever = get_vector_store().as_retriever()
retrieval_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

tools = [
    Tool.from_function(
        name="Resume Feedback",
        func=resume_feedback,
        description="Provides personalized feedback on your resume content"
    ),
    Tool.from_function(
        name="Job Matcher",
        func=job_matcher,
        description="Suggests jobs based on your skills"
    ),
    Tool.from_function(
        name="Mock Interview Response",
        func=mock_interview_response,
        description="Provides a mock answer to an interview question"
    ),
    Tool(
        name="Resume Retriever",
        func=retrieval_chain.run,
        description="Looks up info from your uploaded resume"
    )
]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    memory=memory,
    agent="chat-conversational-react-description",
    verbose=True
)

def format_response_markdown(response):
    # Safely handle line breaks
    formatted_response = response.strip().replace('\n', '\n\n')
    formatted = f"""### 💡 AI Response

{formatted_response}

---

✅ *Need more help? Just ask another question.*
"""
    return formatted

def run_agent(user_input):
    response = agent.run(user_input)
    return format_response_markdown(response)
