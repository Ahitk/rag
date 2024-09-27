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
from langchain.docstore.document import Document
from langchain.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.load import dumps, loads
from indexing import get_vectorstore
import prompts
import initials

# Define the directory containing the rag data
data_directory = "/Users/taha/Desktop/rag/data"

# Load API Keys from environment variables
load_dotenv()  # Load environment variables from a .env file

# Retrieve API keys from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize the chat model and embedding model
model = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)
embedding = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

# Get answer
def get_response(user_input, chat_history):
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

    # Now, use the generated queries to retrieve documents
    if multiple_queries:
        # Use retriever to fetch documents for each query
        documents = []
        for query in multiple_queries:
            retrieved_docs = retriever.get_relevant_documents(query)
            documents.append(retrieved_docs)

        # Use the get_unique_union function to ensure unique documents
        multi_query_docs = get_unique_union(documents)

    # Create prompt for final response generation
    multi_query_rag_chain = (prompts.prompt_telekom | model | StrOutputParser())

    # Generate the final response using RAG with retrieved documents and user question
    response = multi_query_rag_chain.stream({
        "context": multi_query_docs, 
        "question": user_input,
        "chat_history": chat_history
    }) if multi_query_docs else "No relevant documents found."

    # Return both the response and the generated multiple queries for display
    return response, multiple_queries, initials.format_docs(multi_query_docs, user_input)

# Streamlit UI section
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I am a Telekom IT Support chatbot. How can I help you?"),
    ]

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
if user_query:
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        # Get the response, generated queries, and retrieved documents
        response, queries, documents = get_response(user_query, st.session_state.chat_history)

        # Display the AI's response using st.write
        st.write_stream(response)

    with st.sidebar:
        # List the generated queries below the response
        st.markdown("### Ähnliche Fragen:")
        for idx, query in enumerate(queries, start=1):
            st.write(f"{idx}. {query}")  
    
        st.markdown("### Abgerufene Dokumente:")
        st.write(documents)
            
    # Append the AI response to the session state chat history
    st.session_state.chat_history.append(AIMessage(response))