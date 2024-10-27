import time
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
import graph_fusion
from initials import prune_chat_history_if_needed

# Initialize the chat history and token/cost tracking
try:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "question_history" not in st.session_state:
        st.session_state.question_history = []
    if "total_tokens" not in st.session_state:
        st.session_state.total_tokens = 0
    if "total_cost" not in st.session_state:
        st.session_state.total_cost = 0.0
    if "sidebar_shown" not in st.session_state:  # Sidebar hidden initially
        st.session_state.sidebar_shown = False
except Exception:
    st.warning("An error occurred: Please refresh the page.", icon="⚠️")

# Set up the page
try:
    st.set_page_config(page_title="Telekom Help Bot")
    st.image("telekom.png")
except Exception:
    st.warning("An error occurred: Please refresh the page.", icon="⚠️")

# Start chat and display chat history
try:
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.markdown(message.content)
        else:
            with st.chat_message("AI"):
                st.markdown(message.content)
except Exception:
    st.warning("An error occurred: Please refresh the page.", icon="⚠️")

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
                # Get the response, generated queries, and retrieved documents
                total_tokens = st.session_state.total_tokens
                total_cost = st.session_state.total_cost

                chat_history = st.session_state.chat_history
                question_history = st.session_state.question_history

                try:
                    response, documents, tokens, cost = graph_fusion.run_graph(
                        user_query, chat_history, question_history, total_tokens, total_cost, []
                    )
                except Exception:
                    st.warning("An error occurred while processing your request.", icon="⚠️")
                    documents = []  # Ensure documents is empty in case of an error

                # Update token and cost information
                st.session_state.total_tokens = tokens
                st.session_state.total_cost = cost

                if response:
                    # Calculate response time
                    response_time = time.time() - start_time
                    st.markdown(f"{response}\n\n**Response time:** {response_time:.1f}s")

                    # Append the AI response to the session state chat history
                    st.session_state.chat_history.append(AIMessage(content=response))
                    st.session_state.question_history.append(HumanMessage(content=user_query))

                    # Prune chat history before processing
                    prune_chat_history_if_needed()

                    # Show the sidebar only after the first response
                    st.session_state.sidebar_shown = True
    except Exception:
        st.warning("Please refresh the page.", icon="⚠️")

# Sidebar with token count and retrieved documents, only after first response
if st.session_state.sidebar_shown:
    try:
        with st.sidebar:
            st.markdown(f"### Total Chat Token Count: {st.session_state.total_tokens}")
            st.markdown(f"### Total Chat Cost (USD): ${st.session_state.total_cost:.6f}")

            st.markdown("### Retrieved documents:")
            if 'documents' in locals():
                st.write(documents)
            else:
                st.write("No documents retrieved.")
    except Exception:
        st.warning("An error occurred: Please refresh the page.", icon="⚠️")