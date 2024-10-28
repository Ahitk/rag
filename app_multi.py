import time
import streamlit as st
import graph
import chromadb
from langchain_core.messages import AIMessage, HumanMessage
from initials import prune_chat_history_if_needed

# Initialize the chat history and question history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "question_history" not in st.session_state:
    st.session_state.question_history = []

# Initialize documents as an empty list
documents = []

# Set up the page
try:
    st.set_page_config(page_title="Telekom Help Bot")
    st.image("telekom.png")
except Exception as e:
    st.warning(f"An error occurred while setting up the page: {e}", icon="⚠️")

# Start chat and display chat history
try:
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.markdown(message.content)
        else:
            with st.chat_message("AI"):
                st.markdown(message.content)
except Exception as e:
    st.warning(f"An error occurred while displaying chat history: {e}", icon="⚠️")

# User input
user_query = st.chat_input("What would you like to know?")
if user_query:
    try:
        st.session_state.chat_history.append(HumanMessage(content=user_query))

        with st.chat_message("Human"):
            st.markdown(user_query)

        start_time = time.time()  # Start timing
        with st.spinner("In progress..."):
            with st.chat_message("AI"):
                chat_history = st.session_state.chat_history
                question_history = st.session_state.question_history

                try:
                    response, documents = graph.run_graph_multi(user_query, chat_history, question_history, documents)

                    # Move clear_system_cache here to ensure it's called after the response is processed
                    if response:
                        # Append the AI response to the session state chat history
                        st.session_state.chat_history.append(AIMessage(content=response))
                        st.session_state.question_history.append(HumanMessage(content=user_query))

                        # Clear the system cache after processing the response
                        chromadb.api.client.SharedSystemClient.clear_system_cache()

                        # Calculate response time
                        response_time = time.time() - start_time

                        # Display the AI's response with the response time
                        st.markdown(f"{response}\n\n**Response time:** {response_time:.1f}s")

                        # Prune chat history before processing
                        prune_chat_history_if_needed()

                except Exception as e:
                    st.warning(f"An error occurred while running the graph: {e}.\nPlease refresh the page.", icon="⚠️")
                    documents = []  # Ensure documents is empty in case of an error

    except Exception as e:
        st.warning(f"An error occurred while processing your input: {e}.\nPlease refresh the page.", icon="⚠️")

# Display retrieved documents in the sidebar only after user input
if user_query:
    try:
        with st.sidebar:
            st.markdown("### Retrieved documents:")
            if documents:  # Only display documents if they are defined
                st.write(documents)
            else:
                st.write("No documents retrieved.")
    except Exception as e:
        st.warning(f"An error occurred while displaying documents: {e}.\nPlease refresh the page.", icon="⚠️")