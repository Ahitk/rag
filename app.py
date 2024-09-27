# Import necessary libraries
import os
import numpy as np
import glob
import gc
import tiktoken
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.docstore.document import Document
from langchain.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.summarize import load_summarize_chain
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate, FewShotChatMessagePromptTemplate, PromptTemplate
from langchain.load import dumps, loads
from langchain.schema import Document
from langgraph.graph import END, StateGraph, START
from operator import itemgetter
from tavily import TavilyClient
from typing import Literal, List, Tuple
from typing_extensions import TypedDict
from pprint import pprint
from indexing import get_vectorstore
import prompts

# Define the directory containing the rag data
data_directory = "/Users/taha/Desktop/rag/data"

# Load API Keys from environment variables
load_dotenv()  # Load environment variables from a .env file

# Retrieve API keys from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Initialize the chat model and embedding model
# ChatOpenAI is used to interact with the OpenAI GPT model, and OpenAIEmbeddings is used for generating embeddings for documents
model = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)
embedding = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

# Asynchronous function to print generated queries
def print_generated_queries(question, multi_query_chain):
    """
    Generates and prints multiple search queries related to the input question.
    
    Parameters:
    - question (str): The input query for which related search queries are generated.
    """
    multiple_queries = multi_query_chain.stream({"question": question})
    print("\nGenerated Questions:")
    for q in multiple_queries:
        print(f"{q}")

# Get answer
def get_response(user_input):
    # Load vector store and retriever
    vector_store = get_vectorstore(user_input, model, data_directory, embedding)
    retriever = vector_store.as_retriever()

    # Generate multiple queries using the multi_query_prompt and model
    generate_multi_queries = (
        prompts.multi_query_prompt 
        | model 
        | StrOutputParser() 
        | (lambda x: x.split("\n"))  # Split the generated output into individual queries
    )

    # Retrieve and return unique documents
    def get_unique_union(documents):
        """
        Returns a unique union of retrieved documents by flattening and removing duplicates.
        """
        flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
        unique_docs = list(set(flattened_docs))  # Remove duplicates
        return [loads(doc) for doc in unique_docs]

    # Generate the multiple queries based on user input
    multiple_queries = generate_multi_queries.invoke({"question": user_input})
    
    # DEBUG: Check if multiple queries are generated
    print("Generated Queries:", multiple_queries)

    # Now, use the generated queries to retrieve documents
    # Check if multiple_queries is not empty
    if multiple_queries:
        # Use retriever to fetch documents for each query
        documents = []
        for query in multiple_queries:
            retrieved_docs = retriever.get_relevant_documents(query)
            documents.append(retrieved_docs)

        # Use the get_unique_union function to ensure unique documents
        multi_query_docs = get_unique_union(documents)

    # DEBUG: Check if documents are retrieved
    print("Retrieved Documents:", multi_query_docs)

    # Create prompt for final response generation
    multi_query_rag_chain = (prompts.prompt_telekom | model | StrOutputParser())

    # Generate the final response using RAG with retrieved documents and user question
    if multi_query_docs:
        response = multi_query_rag_chain.stream({
            "context": multi_query_docs, 
            "question": user_input,
            "chat_history": st.session_state.chat_history
        })

    # Return both the response and the generated multiple queries for display
    return response, multiple_queries

# Streamlit UI section
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I am a Telekom IT Support chatbot. How can I help you?"),
    ]

if "vector_store" not in st.session_state:
    st.session_state.vector_store = []    

st.set_page_config(page_title="Telekom Hilfe Bot")
st.image("telekom.png")

with st.chat_message(name="assistant"):
    st.write("Hallo! Ich möchte Ihnen bei Ihren Anliegen helfen.")

# Display chat history
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)
    else:
        with st.chat_message("AI"):
            st.markdown(message.content)

# User input
user_query = st.chat_input("Was möchten Sie wissen?")
if user_query is not None and user_query != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    # Display user message in chat message container
    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        # Get the response and generated queries
        response, queries = get_response(user_query)
        
        # Display the AI's response
        #st.markdown(response)
        response = st.r(response)
        
        # List the generated queries below the response
        st.markdown("### Benzer Sorular:")
        for idx, query in enumerate(queries, start=1):
            st.markdown(f"{idx}. {query}")
            
    # Append the AI response to the session state chat history
    st.session_state.chat_history.append(AIMessage(response))