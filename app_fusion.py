import time
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
import graph_fusion

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
user_query = st.chat_input("What would you like to know?")
if user_query:
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    #user_query = "\n".join([msg.content for msg in st.session_state.question_history])
    
    with st.chat_message("Human"):
        st.markdown(user_query)

    start_time = time.time()  # Start timing
    with st.spinner("In progress..."):
        with st.chat_message("AI"):
            # Get the response, generated queries, and retrieved documents
            total_tokens = st.session_state.total_tokens
            total_cost = st.session_state.total_cost
            documents = []
            
            chat_history = st.session_state.chat_history
            question_history = st.session_state.question_history

            response, documents, tokens, cost = graph_fusion.run_graph(user_query, chat_history, question_history, total_tokens, total_cost, documents)

            st.session_state.total_tokens = tokens
            st.session_state.total_cost = cost
            if response:
                # Calculate response time
                response_time = time.time() - start_time

                # Display the AI's response with the response time
                st.markdown(f"{response}\n\n**Response time:** {response_time:.1f}s")

                # Append the AI response to the session state chat history

                st.session_state.chat_history.append(AIMessage(content=response))
                st.session_state.question_history.append(HumanMessage(content=user_query))

    with st.sidebar:
        # Display the token count in the sidebar
        st.markdown(f"### Total Chat Token Count: {st.session_state.total_tokens}")
        st.markdown(f"### Total Chat Cost (USD): ${st.session_state.total_cost:.6f}") 
    
        st.markdown("### Retrieved documents:")
        st.write(documents)