import io
import os
import PIL.Image as Image

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing_extensions import Annotated, Literal, TypedDict

from langchain_core.messages import (
    AnyMessage,
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    trim_messages
)
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate 
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Command


load_dotenv()
os.environ["LANGCHAIN_PROJECT"] = "sales-supervisor"

model = "qwen2.5:14b"
ctx_size = 4000
k_docs = 3
k_mmr = 10
k_limit = 10
workers = ["sales_assistant", "sql_expert"]
options = ["FINISH"] + workers

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    goto: Literal[*options]
    context: str
    input: str
    # answer: str

class SQLQuery(TypedDict):
    """Generated SQL query."""
    query: Annotated[str, ..., "Syntactically valid SQL query."]

class SubmitAnswer(BaseModel):
    """Submit the final answer to the user based only on the query results."""
    answer: str = Field(..., description="The final answer to the user")

llm = ChatOllama(model=model, temperature=0)

### SQL Database
db = SQLDatabase.from_uri("sqlite:///data/tyres.db")
db_ctx = db.get_context()
dialect = db.dialect
# dialect = "SQLite"
table_names = ", ".join(db.get_usable_table_names())
# db.run("SELECT * FROM pricelist;")

toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()
sql_db_query = tools[0]
# print([tool.name for tool in tools])

### Vectorstore

docpath = "data/info.md"
with open(docpath, "r") as file:
    info = file.read()

headers_split = [
    ("#", "Header 1"),
    ("##", "Header 2")
]
md_splitter = MarkdownHeaderTextSplitter(headers_split)
splits = md_splitter.split_text(info)
# TODO: Recursive splitting if chunks are too big

embeddings = OllamaEmbeddings(model=model)
chroma_path = "data/chroma/"

# # Create vectorstore
# vectorstore = Chroma.from_documents(
#     documents=splits,
#     embedding=embeddings,
#     persist_directory=chroma_path
# )
# print("Vector store generated.")

# Load persisted chromadb
vectorstore = Chroma(
    embedding_function=embeddings,
    persist_directory=chroma_path
)
print("Vector store loaded.")

##### Supervisor router #####

# Each worker will perform a task and respond with their results and status.
supervisor_prompt = """You are a sales supervisor from a tyre workshop tasked with answering customer enquiries with the help of the following workers: {workers}. Given the following user enquiry, respond with the worker to act next. If you are able to answer the enquiry, respond with 'FINISH'.
Workers:
1. **sales_assistant**: Use this worker for help in answering enquiries that involve:
    - Store location
    - Opening hours
    - Payment methods 
2. **sql_expert**: Use this worker for help in answering enquiries that involve:
    - Price of tyres 
    - Whether tyres are in stock
    - Handling enquiries that require specific data analysis, calculations, or retrieval from structured SQL databases, particularly when it involves tyres 
Decision Criteria:
- If the enquiry is regarding business information about the tyre workshop, direct it to the **sales_assistant**
- If the enquiry involves structured data points related to tyres and requires a calculation or database lookup, direct it to the **sql_expert**"""
# TODO: FewShotTemplate
# examples_prompt = """Example Decision Flow:
# - Query: "What is the price of the Yokohama BluEarth tyre?" -> **sql_expert**
# - Query: "Is the Pirelli P Zero tyre in stock?" -> **sql_expert**
# - Query: "What are the payment modes available?" -> **sales_assistant**"""
system_prompt = """Given the conversation above, who should act next? Select one of: {options}. If there is enough information to answer the enquiry: "{input}", or if no information is available from the databases or context, select FINISH"""
cpt = ChatPromptTemplate.from_messages([
    ("system", supervisor_prompt),
    ("placeholder", "{messages}"),
    ("system", system_prompt),
]).partial(options=str(options), workers=", ".join(workers))
supervisor_chain = cpt | llm
# TODO: replace END with sales_rep
def supervisor(state: State) -> Command[Literal[*workers, "sales_rep"]]:
    last_message = state["messages"][-1]
    # If last message was the SubmitAnswer tool_call, extract answer
    if getattr(last_message, "tool_calls", None):
        answer = last_message.tool_calls[0]["args"]["answer"]
        state["messages"][-1] = AIMessage(answer)
    response = supervisor_chain.invoke(state)
    goto = response.content
    if len(goto.split()) > 1:
        print(goto)
        raise ValueError("Response has more than one word!")
    if goto == "FINISH":
        goto = "sales_rep"
    # Eliminates the need for conditional edge
    return Command(goto=goto, update={"goto": goto})

sales_rep_jd = "You are a sales representative from KT Tyres tasked with answering customer enquiries."
sales_rep_cpt = ChatPromptTemplate.from_messages([
    ("system", sales_rep_jd),
    ("placeholder", "{messages}"),
])
sales_rep_chain = sales_rep_cpt | llm
def sales_rep(state: State):
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage):
        return state 
    response = sales_rep_chain.invoke(state)
    return {"messages": [response]}

##### Sales assistant (RAG) #####

# Retrieval
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": k_docs, # final no. of docs returned as context
        "fetch_k": k_mmr, # no. of docs to pass to MMR algorithm
    }
)

# Use LLM to rephrase question using chat history as context
# Rephrased query used to retrieve relevant document chunks
contextualize_qn_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)
contextualize_qn_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_qn_system_prompt),
    ("placeholder", "{messages}"),
    ("human", "{input}"),
])
# history_aware_retriever is a runnable that requires input keys: (input,
# messages) and outputs keys: (input, context, answer)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_qn_prompt
)

##### Q&A #####
system_prompt = (
    "You are a sales assistant from KT Tyres. "
    "Use the following pieces of information to answer the question at the end. "
    "If you don't know the answer, just say that you don't know. "
    "Keep the answer as concise as possible.\n"
    "{context}"
)
# System prompt requires provison of context
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("placeholder", "{messages}"),
    # TODO: Check if need to put question at the end
    # ("human", "{input}"),
])
qa_chain = create_stuff_documents_chain(llm, prompt)
# rag_chain is a runnable requiring input keys: (question, messages) and
# outputs keys: (input, messages, context, response)
rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

# Trim chat history
trimmer = trim_messages(
    max_tokens=ctx_size,
    strategy="last",
    token_counter=llm,
    include_system=True,
    allow_partial=False,
    start_on="human",
)

def sales_assistant(state: State):
    # Obtain full history from state and trim
    # Chat history is trimmed before being fed to LLM
    # State still holds entire chat history
    trimmed_history = trimmer.invoke(state["messages"])
    # Trim chat history in state
    # state["messages"] = trimmed_history
    # response = rag_chain.invoke(state)
    response = rag_chain.invoke({
        "input": state["input"],
        "messages": trimmed_history
    })
    return {
        "messages": [("ai", response["answer"])],
        "context": response["context"],
        # "answer": response["answer"],
    }

##### SQL expert #####

# DO NOT call any tool besides SubmitFinalAnswer to submit the final answer.
# DO NOT make stuff up if you don't have enough information to answer the query, just say you don't have enough information.
# Call the SubmitAnswer tool to submit the final answer to the user. 
# DO NOT conclude that there is no entry in the database unless at least one attempt to query the database has been made.
# DO NOT call the SubmitAnswer tool to submit the final answer to the user if you do not have enough information to answer the input question.
agent_template = """You are a SQL expert with a strong attention to detail. Given a user enquiry, output a syntactically correct {dialect} query to run. Only query from tables present in the schema. Pay attention to use only the column names that you can see in the schema. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
Database schema:
{schema}
"""
query_guidelines = """When generating the query:
- Output only the {dialect} query and nothing else
- Keep the query short and simple
- Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most {k_limit} results.
- You can order the results by a relevant column to return the most interesting examples in the database.
- DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.
"""
route_template = """Given the above conversation, output a syntactically correct {dialect} query to run. If you get an error while executing a query, rewrite the query and try again. If you get an empty result set, rewrite the query to get a non-empty result set. If you do not have enough information to answer the input question, write a {dialect} query to retrieve the additional information needed. If you have enough information to answer the initial enquiry: "{input}", call the SubmitAnswer tool to submit your answer."""
template = '\n'.join([agent_template, query_guidelines])
query_cpt = ChatPromptTemplate.from_messages([
    ("system", template),
    ("placeholder", "{messages}"),
    ("system", route_template),
])
generate_agent = query_cpt | llm.bind_tools([SubmitAnswer])
def generate_query(state: State):
    inputs = {
        "dialect": dialect, 
        "schema": db_ctx["table_info"], 
        "k_limit": k_limit,
    }
    inputs.update(state)
    response = generate_agent.invoke(inputs)
    return {"messages": [response]}

# DO NOT return your review, ONLY return output a syntactically correct {dialect} query, nothing else.
correct_template = """You are a SQL expert with a strong attention to detail. Double check the SQL query below for common mistakes. If there are mistakes, call the sql_db_query tool with the corrected query. If there are no mistakes, call the sql_db_query tool with the same query. Ensure that there are no DML statements (INSERT, UPDATE, DELETE, DROP etc.) in the query.
Examples of common mistakes:
- Using NOT IN with NULL values
- Using UNION when UNION ALL should have been used
- Using BETWEEN for exclusive ranges
- Data type mismatch in predicates
- Properly quoting identifiers
- Using the correct number of arguments for functions
- Casting to the correct data type
- Using the proper columns for joins
"""
query_template = """Query:
{query}"""
correct_pt = PromptTemplate.from_template(correct_template)
correct_cpt = ChatPromptTemplate.from_messages([
    ("system", correct_template),
    ("ai", query_template),  
    # ("placeholder", "{messages}"),
])
correct_chain = correct_cpt | llm.bind_tools([sql_db_query], tool_choice="sql_db_query")
def correct_query(state: State):
    last_message = state["messages"][-1]
    message = correct_chain.invoke({
        # "messages": [last_message],
        "query": last_message.content,
        "dialect": dialect
    })
    # TODO: Final check to see that there is no DML statements
    return {"messages": [message]}

def should_continue(state: State) -> Literal["supervisor", "correct_query"]:
    last_message = state["messages"][-1]
    # If there is a tool call, then we finish
    if getattr(last_message, "tool_calls", None):
        return "supervisor" 
    elif last_message.content == "":
        # If no tool call and query is empty
        print("Warning: No response")
        return "supervisor"
    else:
        return "correct_query"

##### Graph #####

workflow = StateGraph(state_schema=State)
workflow.add_node("supervisor", supervisor)
workflow.add_node("sales_assistant", sales_assistant)
workflow.add_node("sql_expert", generate_query)
workflow.add_node("correct_query", correct_query)
workflow.add_node("execute_query", ToolNode([sql_db_query]))
workflow.add_node("sales_rep", sales_rep)
workflow.add_edge(START, "supervisor")
workflow.add_conditional_edges("sql_expert", should_continue)
workflow.add_edge("correct_query", "execute_query")
workflow.add_edge("execute_query", "sql_expert")
workflow.add_edge("sales_assistant", "supervisor")
workflow.add_edge("sales_rep", END)
checkpointer = MemorySaver()
graph = workflow.compile(checkpointer=checkpointer)

user_id = "1"
config = {"configurable": {"thread_id": user_id}}

q1 = "How much for Michelin Energy XM2 tyre?"
q1a = "How much for Yokohama BluEarth tyre?"
q2 = "What time do you close on Monday?"
q3 = "Who are you?"

question = q2
inputs = {
    "messages": [("user", question)],
    "input": question 
}
for step in graph.stream(inputs, config=config, stream_mode="values"):
    step["messages"][-1].pretty_print()

# response = graph.invoke(inputs, config=config)

state = graph.get_state(config)
print(state.values.keys())
messages = state.values['messages']
# Error caused key to be not saved
for message in messages:
    message.pretty_print()

# TODO: Warning: Sometimes sql_expert does not call SubmitAnswer
# TODO: Check RAG end message

# # Visualise subgraph 
# img_bytes = graph.get_graph().draw_mermaid_png()
# image = Image.open(io.BytesIO(img_bytes))
# image = image.resize((1100, 1000), Image.LANCZOS)
# image.show()
# image.save("assets/supervisor.png")
