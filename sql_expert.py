from pydantic import BaseModel, Field
from typing_extensions import Annotated, Literal, TypedDict

from langchain_core.messages import (
    AnyMessage, HumanMessage, SystemMessage, ToolMessage
)
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate 
from langchain_core.tools import tool 
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langchain_ollama import ChatOllama

from langgraph.prebuilt import ToolNode
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
