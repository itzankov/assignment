from fastapi import FastAPI
from pydantic import BaseModel
import os
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain.tools import tool
from langchain.callbacks.tracers.langchain import LangChainTracer

# Load environment variables
os.environ["LANGCHAIN_API_KEY"] = ""
os.environ["LANGCHAIN_PROJECT"] = "BI_assignment"
os.environ["OPENAI_API_KEY"] = ""

app = FastAPI()

# Define a simple tool
@tool
def life_expectancy(expression: age) -> str:
    """Evaluates a basic math expression."""
    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"

# Initialize LangChain agent with LangSmith tracing
llm = ChatOpenAI(temperature=0)
tools = [life_expectancy]
tracer = LangChainTracer(project_name=os.environ["LANGCHAIN_PROJECT"])
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, callbacks=[tracer])

# Request model
class UserInput(BaseModel):
    message: str

# Endpoint
