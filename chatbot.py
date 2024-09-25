import os
import streamlit as st
from openai import OpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# Set up your OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
model = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.set_page_config(page_title="Telekom Hilfe Bot") #page_icon ekleyebilirsin parantez icine ama bakmadim nasil calisiyor.
st.image("telekom.png")
#st.title('Telekom Hilfe Chatbot')

with st.chat_message(name="assistant"):
    st.write("Hallo! Ich möchte Ihnen bei Ihren Anliegen helfen.")

# Get answer
def get_answer(question, chat_history):
    telekom_template = """You are a helpful assistant. Answer the following questions considering the history of the conversation:
    Question: {question}
    Context: {chat_history}
    Answer:
    """
    prompt_telekom = ChatPromptTemplate.from_template(telekom_template)

    rag_chain = (prompt_telekom | model | StrOutputParser())
    #return rag_chain.invoke({"question": question, "chat_history": chat_history})
    return rag_chain.stream({"question": question, "chat_history": chat_history}) 

# Conversation
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)
    else:
        with st.chat_message("AI"):
            st.markdown(message.content)

# User Input
user_query = st.chat_input("Was möchten Sie wissen?")
if user_query is not None and user_query != "":
    st.session_state.chat_history.append(HumanMessage(user_query))
    # Display user message in chat message container
    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        #ai_response = get_answer(user_query, st.session_state.chat_history)
        #st.markdown(ai_response)
        ai_response = st.write_stream(get_answer(user_query, st.session_state.chat_history))

    st.session_state.chat_history.append(AIMessage(ai_response))
   