import time
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
import graph_fusion
from initials import prune_chat_history_if_needed

# Initialize the chat history and token/cost tracking
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "question_history" not in st.session_state:
    st.session_state.question_history = []

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
            documents = []
            
            chat_history = st.session_state.chat_history
            question_history = st.session_state.question_history

            response, documents= graph_fusion.run_graph(user_query, chat_history, question_history, documents)
  
            if response:
                # Calculate response time
                response_time = time.time() - start_time

                # Display the AI's response with the response time
                st.markdown(f"{response}\n\n**Response time:** {response_time:.1f}s")

                # Append the AI response to the session state chat history

                st.session_state.chat_history.append(AIMessage(content=response))
                st.session_state.question_history.append(HumanMessage(content=user_query))
                # Prune chat history before processing
            prune_chat_history_if_needed()

    with st.sidebar:
        st.markdown("### Retrieved documents:")
        st.write(documents)