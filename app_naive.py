# Import necessary libraries
import os
import time  # Import time to measure response time
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import DirectoryLoader, TextLoader
import prompts as prompts
import initials as initials
from langchain_community.callbacks import get_openai_callback

# Initialize the chat history and token/cost tracking
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "question_history" not in st.session_state:
    st.session_state.question_history = []
if "total_tokens" not in st.session_state:
    st.session_state.total_tokens = 0
if "total_cost" not in st.session_state:
    st.session_state.total_cost = 0.0

st.set_page_config(page_title="Telekom Help Bot")
st.image("telekom.png")

# Function to get response with error handling
def get_response(user_input, chat_history):
    try:
        # Prune chat history before processing
        initials.prune_chat_history_if_needed()

        # Load all .txt files from the specified directory using DirectoryLoader
        loader = DirectoryLoader(initials.data_directory, glob="**/*.txt", loader_cls=TextLoader)

        # Load documents
        docs = loader.load()

        # Embedding
        vectorstore = Chroma.from_documents(documents=docs, embedding=initials.embedding)
        retriever = vectorstore.as_retriever(search_kwargs={'k': 10})
        retrieved_docs = retriever.invoke(user_input)

        # Create prompt for final response generation
        rag_chain = (prompts.prompt_telekom | initials.model | StrOutputParser())

        # Use OpenAI callback to track costs and tokens
        with get_openai_callback() as cb:
            response = rag_chain.invoke({
                "context": retrieved_docs, 
                "question": user_input,
                "chat_history": chat_history
            }) if retrieved_docs else "No relevant documents found."

        # Update total tokens and cost
        st.session_state.total_tokens += cb.total_tokens
        st.session_state.total_cost += cb.total_cost

        # Return both the response, the generated multiple queries, and the retrieved documents
        return response, initials.format_documents(retrieved_docs, user_input)

    except FileNotFoundError:
        st.error("Documents could not be loaded. Please check the data directory path.")
        return None, None, None

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return None, None, None


# Start chat
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

    start_time = time.time()  # Start timing
    with st.spinner("In progress..."):  # Spinner to show processing
        with st.chat_message("AI"):
            # Get the response, generated queries, and retrieved documents
            response, documents = get_response(user_query, st.session_state.chat_history)
            print("==========   PROCESS ENDED  ==========")
            # Calculate response time
            response_time = time.time() - start_time

            # Display the AI's response with the response time
            st.markdown(f"{response}\n\n**Response time:** {response_time:.1f}s")

    # Append the AI response to the session state chat history
    
    st.session_state.chat_history.append(AIMessage(content=response))

    with st.sidebar:
        # Display the token count in the sidebar
        st.markdown(f"### Total Chat Token Count: {st.session_state.total_tokens}")
        st.markdown(f"### Total Chat Cost (USD): ${st.session_state.total_cost:.6f}")  # Display total cost        
        st.markdown("### Retrieved documents:")
        st.write(documents)