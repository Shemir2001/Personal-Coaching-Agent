from langchain.agents import initialize_agent, Tool
from langchain_community.chat_models import ChatGoogleGenerativeAI
from tools import resume_feedback, job_matcher, mock_interview_response
from memory import get_memory
from vector_store import get_vectorstore
from langchain.chains import RetrievalQA

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

memory = get_memory()

retriever = get_vectorstore().as_retriever()
retrieval_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

tools = [
    Tool.from_function(func=resume_feedback),
    Tool.from_function(func=job_matcher),
    Tool.from_function(func=mock_interview_response),
    Tool(name="Resume Retriever", func=retrieval_chain.run, description="Looks up info from your uploaded resume")
]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    memory=memory,
    agent="chat-conversational-react-description",
    verbose=True
)

def run_agent(user_input):
    return agent.run(user_input)
