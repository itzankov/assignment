from fastapi import FastAPI
import os
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent
from langchain.agents.agent_types import AgentType
from langchain.tools import tool
from langchain.callbacks.tracers.langchain import LangChainTracer
import train
import pandas as pd

recomender_model = train.train_model()

life_expectancy_data = pd.read_csv("life_expectancy.csv")

# Load environment variables
os.environ["LANGCHAIN_API_KEY"] = ""
os.environ["LANGCHAIN_PROJECT"] = "BI_assignment"
os.environ["OPENAI_API_KEY"] = ""

app = FastAPI()

def numeric_sex(sex):
    if sex.lower()[0] == "f":
        return 0
    else:
        return 1

# Define a simple tool
@tool
def recommend_tv_show(age: int, sex: str) -> str:
    """recommend tv show based on age and sex"""
    try:
        result = recomender_model.predict([age, numeric_sex(sex)]);
        return f"Result: {result[0]}"
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def life_expectancy(age: int, sex: str) -> str:
    """Alculate life expectancy based on age and sex"""
    try:
        if age < 0 or age > 119:
            return f"age must be between 0 and 119"
        age_index = age;
        if not numeric_sex(sex):
            age_index += 120
        return life_expectancy_data.iloc[age_index][4]
    except Exception as e:
        return f"Error: {str(e)}"

# Initialize LangChain agent with LangSmith tracing
llm = ChatOpenAI(temperature=0)
tools = [recommend_tv_show, life_expectancy]
tracer = LangChainTracer(project_name=os.environ["LANGCHAIN_PROJECT"])
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, callbacks=[tracer])

# Endpoint
