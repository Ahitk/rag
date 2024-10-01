# Import necessary libraries
import os
import time  # Import time to measure response time
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from langchain.load import dumps, loads
from indexing import get_vectorstore
import prompts
import initials
from langchain_community.callbacks import get_openai_callback

# Define the directory containing the rag data
data_directory = "/Users/taha/Desktop/rag/data"

# Load API Keys from environment variables
load_dotenv()  # Load environment variables from a .env file

# Retrieve API keys from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize the chat history and token/cost tracking
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "question_history" not in st.session_state:
    st.session_state.question_history = []
if "total_tokens" not in st.session_state:
    st.session_state.total_tokens = 0  # Track total tokens
if "total_cost" not in st.session_state:
    st.session_state.total_cost = 0.0  # Track total cost
if "model" not in st.session_state:
    st.session_state.model = None  # Track selected model

st.set_page_config(page_title="Telekom Hilfe Bot")
st.image("telekom.png")

# Available models
models = {
    "GPT-4o mini: Affordable and intelligent small model for fast, lightweight tasks": "gpt-4o-mini",
    "GPT-4o: High-intelligence flagship model for complex, multi-step tasks": "gpt-4o",
    "GPT-4: The previous set of high-intelligence model": "gpt-4",
    "GPT-3.5 Turbo: A fast, inexpensive model for simple tasks": "gpt-3.5-turbo-0125",
}

# Function to get answer
def get_response(user_input, chat_history, question_history):
    # Load vector store and retriever
    vector_store = get_vectorstore(user_input, model, data_directory, embedding)
    retriever = vector_store.as_retriever()

    # Generate step-back queries
    generate_stepback_question = prompts.step_back_prompt | model | StrOutputParser()
    step_back_question = generate_stepback_question.invoke({"question": user_input, "question_history": question_history })

    docs = initials.format_docs(retriever.invoke(user_input), user_input)
    stepback_docs = initials.format_docs(retriever.invoke(step_back_question), user_input)

    step_back_chain = (
    {
        "chat_history": lambda x: x["chat_history"],
        "normal_context": lambda x: initials.format_docs(retriever.invoke(x["question"]), (x["question"]) ),
        "question": lambda x: x["question"],
        "step_back_context": lambda x: initials.format_docs(retriever.invoke(x["step_back_question"]), (x["step_back_question"])),
        "question_history": lambda x: x["question_history"],
    }
    | prompts.stepback_response_prompt
    | model
    | StrOutputParser()
    )

    # Use OpenAI callback to track costs and tokens
    with get_openai_callback() as cb:
        response = step_back_chain.invoke({
            "question": user_input,
            "step_back_question": step_back_question,
            "chat_history": chat_history,
            "question_history": question_history
        })


    # Update total tokens and cost
    st.session_state.total_tokens += cb.total_tokens
    st.session_state.total_cost += cb.total_cost

    # Return both the response, the generated multiple queries, and the retrieved documents
    return response, step_back_question, docs, stepback_docs


# Dropdown for selecting model (only if a model hasn't been selected yet)
if st.session_state.model is None:
    selected_model = st.selectbox("Select the OpenAI model to use:", list(models.keys()), index=None, placeholder="...")

    if selected_model:  # Ensure a model has been selected
        # Update selected model in session state
        st.session_state.model = models[selected_model]

        # Initialize the model and embedding based on the selected model
        model = ChatOpenAI(model=st.session_state.model, api_key=OPENAI_API_KEY)
        embedding = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

        st.write("Model selected! Start chatting below.")
else:
    # Model is already selected
    model = ChatOpenAI(model=st.session_state.model, api_key=OPENAI_API_KEY)
    embedding = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

# Start chat if model has been selected
if st.session_state.model:
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
                response, stepback_query, documents, stepback_documents = get_response(user_query, st.session_state.chat_history, st.session_state.question_history)

                # Calculate response time
                response_time = time.time() - start_time

                # Display the AI's response with the response time
                st.markdown(f"{response}\n\n**Response time:** {response_time:.1f}s")

        # Append the AI response to the session state chat history
       
        st.session_state.chat_history.append(AIMessage(content=response))
        st.session_state.question_history.append(HumanMessage(content=user_query))
        st.session_state.question_history.append(HumanMessage(content=stepback_query))

        with st.sidebar:
            # Display the selected model short name (key) at the top of the sidebar
            st.markdown(f"### Selected Model: {st.session_state.model}")  # Now shows only the model key
            # Display the token count in the sidebar
            st.markdown(f"### Total Chat Token Count: {st.session_state.total_tokens}")
            st.markdown(f"### Total Chat Cost (USD): ${st.session_state.total_cost:.6f}")  # Display total cost

            st.markdown("### Step-back question:")
            st.markdown(stepback_query) 

            st.markdown("### Step-back documents:")
            st.write(stepback_documents)

            st.markdown("### Normal documents:")
            st.markdown(documents) 
        
