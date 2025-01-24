#!/usr/bin/env python 

import os
import sqlite3
from dotenv import load_dotenv
from typing import Sequence
from typing_extensions import Annotated, TypedDict

from langchain_core.messages import (
    BaseMessage, AIMessage, HumanMessage, SystemMessage, trim_messages
)
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_chroma import Chroma
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_ollama.llms import OllamaLLM
from langchain_ollama import ChatOllama, OllamaEmbeddings

from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages


# Log trace to LangSmith
load_dotenv()
os.environ["LANGCHAIN_PROJECT"] = "sales-assistant"

ctx_size = 2000
model = "qwen2.5:14b"

##### Documents #####

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

# Create vectorstore
vectordb = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory=chroma_path
)
print("Vector store generated.")

# Load persisted chromadb
vectorstore = Chroma(
    embedding_function=embeddings,
    persist_directory=chroma_path
)
print("Vector store loaded.")

##### Retrieval #####
retriever = vectorstore.as_retriever(
    search_type="mmr", search_kwargs={"k": 3, "n_results": 3}
)
llm = ChatOllama(model=model)

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
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])
# history_aware_retriever is a runnable that requires input keys: (input,
# chat_history) and outputs keys: (input, context, answer)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_qn_prompt
)

##### Q&A #####
system_prompt = (
    "You are Alex, a tyre salesperson from KT Tyres. "
    "Use the following pieces of information to answer the question at the end. "
    "If you don't know the answer, just say that you don't know. "
    "Keep the answer as concise as possible.\n"
    "{context}"
)

# System prompt requires provison of context
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])
# prompt requires context, chat_history, input
# qa_chain is a runnable that requires input keys: (context, chat_history, input)
qa_chain = create_stuff_documents_chain(llm, prompt)
# rag_chain is a runnable requiring input keys: (input, chat_history) and
# outputs keys: (input, chat_history, context, response)
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

##### LangGraph #####

class State(TypedDict):
    input: str
    chat_history: Annotated[Sequence[BaseMessage], add_messages]
    context: str
    answer: str

# Define node in LangGraph, return values are used to update state
def call_model(state: State):
    trimmed_history = trimmer.invoke(state["chat_history"])
    # Trim chat history in state
    # state["chat_history"] = trimmed_history
    # response = rag_chain.invoke(state)
    # Chat history is trimmed before being fed to LLM. State still holds entire
    # chat history.
    response = rag_chain.invoke({
        "input": state["input"],
        "chat_history": trimmed_history
    })
    return {
        "chat_history": [
            HumanMessage(state["input"]),
            AIMessage(response["answer"]),
        ],
        "context": response["context"],
        "answer": response["answer"],
    }

workflow = StateGraph(state_schema=State)
workflow.add_node("model", call_model)
workflow.add_edge(START, "model")
workflow.add_edge("model", END)

# Persist graph in Sqlite database
sqlite_path = "data/langgraph_memory.db"
conn = sqlite3.connect(sqlite_path, check_same_thread=False)
memory = SqliteSaver(conn)
graph = workflow.compile(checkpointer=memory)

# TODO: async langchain support
if __name__ == "__main__":
    print("Enter user ID: ", end="")
    user_id = input()
    config = {"configurable": {"thread_id": user_id}}
    while True:
        print("> ", end="")
        qn = input()
        if qn == "\q":
            print("Quit chat.")
            break
        result = graph.invoke({"input": qn}, config=config)
        print(result["answer"])


# # Manual input
# user_id = "1"
# config = {"configurable": {"thread_id": user_id}}
# 
# qn = "what's my name?"
# result = graph.invoke({"input": qn}, config=config)
# print(result)
# 
# # Check chat history of user ID: 1
# # config = {"configurable": {"thread_id": "1"}}
# state = graph.get_state(config)
# chat_history = state.values["chat_history"]
# for message in chat_history:
#     message.pretty_print()
