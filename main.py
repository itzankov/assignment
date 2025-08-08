from fastapi import FastAPI
from pydantic import BaseModel
import os
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain.tools import tool
from langchain.callbacks.tracers.langchain import LangChainTracer
import train

model = train.train_model()


# Load environment variables
os.environ["LANGCHAIN_API_KEY"] = ""
os.environ["LANGCHAIN_PROJECT"] = "BI_assignment"
os.environ["OPENAI_API_KEY"] = ""

app = FastAPI()

# Define a simple tool
@tool
def recommend(age: int, sex: int) -> str:
    """recommend tv show."""
    try:
        result = model.predict([age, sex]);
        return f"Result: {result[0]}"
    except Exception as e:
        return f"Error: {str(e)}"

# Initialize LangChain agent with LangSmith tracing
llm = ChatOpenAI(temperature=0)
tools = [recommend]
tracer = LangChainTracer(project_name=os.environ["LANGCHAIN_PROJECT"])
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, callbacks=[tracer])

# Request model
class UserInput(BaseModel):
    message: str

# Endpoint
