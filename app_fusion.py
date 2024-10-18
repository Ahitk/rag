# Import necessary libraries
import os
import time  # Import time to measure response time
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from indexing import get_vectorstore
from operator import itemgetter
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

# Function to get answer
def get_response(user_input, chat_history, question_history):
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

    # Generate the multiple queries based on user input
    multiple_queries = generate_multi_queries.invoke({"question": user_input, "question_history": question_history})

    retrieval_chain_rag_fusion = generate_multi_queries | retriever.map() | initials.reciprocal_rank_fusion

    fusion_docs = retrieval_chain_rag_fusion.invoke({"question": user_input, "question_history": question_history})
    formatted_docs = initials.format_fusion_docs_with_similarity(fusion_docs, user_input)

    fusion_rag_chain = (prompts.prompt_telekom | model | StrOutputParser())

  # Use OpenAI callback to track costs and tokens
    with get_openai_callback() as cb:
        response = fusion_rag_chain.invoke({
            "context": formatted_docs, 
            "question": user_input,
            "chat_history": chat_history
        }) if formatted_docs else "No relevant documents found."

    # Update total tokens and cost
    st.session_state.total_tokens += cb.total_tokens
    st.session_state.total_cost += cb.total_cost

    # Return both the response, the generated multiple queries, and the retrieved documents
    return response, multiple_queries, formatted_docs


# Dropdown for selecting model (only if a model hasn't been selected yet)
if st.session_state.model is None:
    selected_model = st.selectbox("Select the OpenAI model to use:", list(initials.models.keys()), index=None, placeholder="...")

    if selected_model:  # Ensure a model has been selected
        # Update selected model in session state
        st.session_state.model = initials.models[selected_model]

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
                response, queries, documents = get_response(user_query, st.session_state.chat_history, st.session_state.question_history)

                # Calculate response time
                response_time = time.time() - start_time

                # Display the AI's response with the response time
                st.markdown(f"{response}\n\n**Response time:** {response_time:.1f}s")

        # Append the AI response to the session state chat history
       
        st.session_state.chat_history.append(AIMessage(content=response))
        st.session_state.question_history.append(HumanMessage(content=queries))

        with st.sidebar:
            # Display the selected model short name (key) at the top of the sidebar
            st.markdown(f"### Selected Model: {st.session_state.model}")  # Now shows only the model key
            # Display the token count in the sidebar
            st.markdown(f"### Total Chat Token Count: {st.session_state.total_tokens}")
            st.markdown(f"### Total Chat Cost (USD): ${st.session_state.total_cost:.6f}")  # Display total cost

            # List the generated queries and retrieved documents in the sidebar
            st.markdown("### Similar questions:")
            for idx, query in enumerate(queries, start=1):
                st.write(f"{idx}. {query}")  
        
            st.markdown("### Retrieved documents:")
            st.write(documents)