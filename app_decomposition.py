# RESPONSE TIME COK UZUN VE RETRIEVED DOCUMENTS GELMIYOR


# Import necessary libraries
import os
import time  # Import time to measure response time
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from operator import itemgetter
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

# Function to get answer
def get_response(user_input, chat_history, question_history):
    # Load vector store and retriever
    vector_store = get_vectorstore(user_input, model, data_directory, embedding)
    retriever = vector_store.as_retriever()

    # Generate sub-questions related to the main question
    generate_queries_decomposition = (prompts.prompt_subquestions | model | StrOutputParser() | (lambda x: x.split("\n")))

    # Generate decomposition questions using question history and chat history
    decomposition_questions = generate_queries_decomposition.invoke({
        "question": user_input, 
        "question_history": question_history, 
        "chat_history": chat_history
    })

    def format_qa_pair(question, answer):
        """Format Q and A pair"""
        formatted_string = f"Question: {question}\nAnswer: {answer}\n"
        return formatted_string.strip()

    q_a_pairs = ""
    all_responses = []
    final_response = ""  # Burada final_response'i bos bir string olarak tanimliyoruz
        
        # Loop through sub-questions and generate responses for each one
    for q in decomposition_questions:
        # Retrieve context for each question
        decomposition_rag_chain = (
            {"context": itemgetter("question") | retriever, 
            "question": itemgetter("question"),
            "q_a_pairs": itemgetter("q_a_pairs"),
            "chat_history": itemgetter("chat_history")}  # Adding chat history to the retrieval chain
            | prompts.decomposition_prompt
            | model
            | StrOutputParser()
        )

        # Generate a response for the sub-question
        sub_response = decomposition_rag_chain.invoke({
            "question": q, 
            "q_a_pairs": q_a_pairs, 
            "chat_history": chat_history  # Including chat history
        })
            
        # Format the Q&A pair and add to the running list
        q_a_pair = format_qa_pair(q, sub_response)
        
        # Append the individual sub-question's response to all_responses
        all_responses.append(q_a_pair)
            
        # Update q_a_pairs with the new Q&A pair
        q_a_pairs = q_a_pairs + "\n---\n" + q_a_pair
            
        # Create a final combined response for the main question
        final_response += "\n".join(all_responses)  # Bu noktada final_response bos bir stringten baslayarak cevaplari birlestirir.
 
    response_chain = (prompts.prompt_telekom | model | StrOutputParser())
    with get_openai_callback() as cb:
        response = response_chain.invoke({
            "context": final_response, 
            "question": user_input,
            "chat_history": chat_history
        }) 
    # Update total tokens and cost
    st.session_state.total_tokens += cb.total_tokens
    st.session_state.total_cost += cb.total_cost

    # Return both the response, the generated multiple queries, and the retrieved documents
    return response, decomposition_questions


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
                response, queries = get_response(user_query, st.session_state.chat_history, st.session_state.question_history)

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
            st.markdown("### Sub-questions:")
            for idx, query in enumerate(queries, start=1):
                st.write(f"{idx}. {query}")  
        