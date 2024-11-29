#!/usr/bin/env python 

from typing import Sequence
from typing_extensions import Annotated, TypedDict

from langchain_core.messages import (
    BaseMessage, AIMessage, HumanMessage, SystemMessage, trim_messages
)
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_ollama.llms import OllamaLLM
from langchain_ollama import OllamaEmbeddings

from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages


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

# TODO: Persist documents database?
embeddings = OllamaEmbeddings(model="llama3")
vectorstore = InMemoryVectorStore.from_documents(
    documents=splits, embedding=embeddings
)
print("Vector store generated.")

##### Retrieval #####
# TODO: Experiment with more retrievers
retriever = vectorstore.as_retriever()
llm = OllamaLLM(model="llama3")
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
docpath = "data/system_prompt.txt"
with open(docpath, "r") as file:
    system_prompt = file.read()

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
ctx_size = 2048 
trimmer = trim_messages(
    max_tokens=ctx_size,
    strategy="last",
    token_counter=llm,
    include_system=True,
    allow_partial=False,
    start_on="human",
)

##### LangGraph #####
# Dictionary representing state of the application
# State has the same input and output keys as `rag_chain`
class State(TypedDict):
    input: str
    chat_history: Annotated[Sequence[BaseMessage], add_messages]
    context: str
    answer: str

# We then define a simple node that runs the `rag_chain`.
# The `return` values of the node update the graph state, so here we just
# update the chat history with the input message and response.
def call_model(state: State):
    trimmed_history = trimmer.invoke(state["chat_history"])
    # TODO: Check whether to update chat_history in this way
    # TODO: Solution: incorporate trimmer before LLM by implementing custom rag chain
    state["chat_history"] = trimmed_history
    response = rag_chain.invoke(state)
    return {
        "chat_history": [
            HumanMessage(state["input"]),
            AIMessage(response["answer"]),
        ],
        "context": response["context"],
        "answer": response["answer"],
    }

# Our graph consists only of one node:
workflow = StateGraph(state_schema=State)
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

# TODO: Persist using external database? 
# Finally, we compile the graph with a checkpointer object.
# This persists the state, in this case in memory.
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# TODO: Support for multiple users?
# TODO: async langchain support
# Each user given a unique thread_id
config = {"configurable": {"thread_id": "abc123"}}

while True:
    print("> ", end="")
    qn = input()
    if qn == "\q":
        print("Quit chat.")
        break
    result = app.invoke({"input": qn}, config=config)
    print(result["answer"])

chat_history = app.get_state(config).values["chat_history"]
for message in chat_history:
    message.pretty_print()

# TODO: Visualisation using LangSmith
# TODO: LangChain Runnables and LCEL
