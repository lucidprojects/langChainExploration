#!/usr/bin/env python
from typing import List, Optional, Generator

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain import hub
from langchain.agents import create_openai_functions_agent
from langchain.agents import AgentExecutor, Tool, ZeroShotAgent, create_react_agent
from langchain.pydantic_v1 import BaseModel, Field, validator
from langchain_core.messages import BaseMessage
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers.json import SimpleJsonOutputParser

from langserve import add_routes
import asyncio

import os
from dotenv import load_dotenv

import uuid

import logging

# Load the OpenAI API key from the .env file or read from environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 1. Load Retriever, embeddings, and vector
loader = PyPDFLoader(
    "https://nurtur.earth/docs/UCLA-IoES-Practicum-SPA-Virtual-Production-Final-Report-2023-short-clean.pdf"
)
pages = loader.load_and_split()

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY
)

vector = FAISS.from_documents(pages, embeddings)

retriever = vector.as_retriever()

# 2. Create Tools
retriever_tool = create_retriever_tool(
    retriever,
    "production_lca_search",
    "Search for information about UCLA loES Virtual vs. Conventional Production for Film and Television: A Comparative Life Cycle Assessment. For all questions, you must use this tool!",
)

search = TavilySearchResults()
tools = [retriever_tool, search]

# 2.5 Create Memory

sessionId = str(uuid.uuid4())

message_history = RedisChatMessageHistory(
    url="redis://localhost:6379/0", ttl=600, session_id=sessionId
)

memory = ConversationBufferMemory(
    memory_key="chat_history", chat_memory=message_history
)


# 3. Create Agent
prompt = hub.pull("hwchase17/react-chat")

# We need to add these input/output schemas because the current AgentExecutor
# is lacking in schemas.

class Input(BaseModel):
    input: str
    chat_history: Optional[List[BaseMessage]] = None

class Output(BaseModel):
    output: str


llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY, model="gpt-3-turbo", temperature=0
)

agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True, memory=memory, handle_parsing_errors=True
)


# 4. App definition
app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple API server using LangChain's Runnable interfaces",
)

# 4.5 Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# 5. Adding chain route

class BaseMessage(BaseModel):
    content: str
    type: str  # Assuming 'type' field exists based on earlier client code examples

    @validator("content", "type")
    def check_non_empty(cls, v):
        if not v:
            raise ValueError("Must not be empty")
        return v


add_routes(
    app,
    agent_executor.with_types(input_type=Input, output_type=Output),
    path="/agent",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
