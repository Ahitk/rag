# Import necessary libraries
import time
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.callbacks import get_openai_callback
from indexing import get_vectorstore, generate_vectorstore_semantic_chunking
import prompts as prompts
import initials as initials
import chromadb

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
def get_response(user_input, chat_history, question_history):
    try:
        # Prune chat history before processing
        initials.prune_chat_history_if_needed()

        # Load vector store and retriever
        vector_store = generate_vectorstore_semantic_chunking(user_input, initials.model, initials.data_directory, initials.embedding)
        retriever = vector_store.as_retriever()

        # Generate multiple queries using the multi_query_prompt and model
        generate_multi_queries = (
            prompts.multi_query_prompt 
            | initials.model 
            | StrOutputParser() 
            | (lambda x: x.split("\n"))  # Split the generated output into individual queries
        )

        # Generate the multiple queries based on user input
        multiple_queries = generate_multi_queries.invoke({"question": user_input, "question_history": question_history})

        # Now, use the generated queries to retrieve documents
        if multiple_queries:
            # Use retriever to fetch documents for each query
            documents = []
            for query in multiple_queries:
                retrieved_docs = retriever.invoke(query)
                documents.append(retrieved_docs)

            # Use the get_unique_union function to ensure unique documents
            multi_query_docs = initials.get_unique_union(documents)

        # Create prompt for final response generation
        multi_query_rag_chain = (prompts.prompt_telekom | initials.model | StrOutputParser())

        # Use OpenAI callback to track costs and tokens
        with get_openai_callback() as cb:
            response = multi_query_rag_chain.invoke({
                "context": multi_query_docs, 
                "question": user_input,
                "chat_history": chat_history
            }) if multi_query_docs else "No relevant documents found."

        # Update total tokens and cost
        st.session_state.total_tokens += cb.total_tokens
        st.session_state.total_cost += cb.total_cost

        # Return both the response, the generated multiple queries, and the retrieved documents
        return response, multiple_queries, initials.format_documents(multi_query_docs, user_input)

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
user_query = st.chat_input("What would you like to know?")
if user_query:
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    
    with st.chat_message("Human"):
        st.markdown(user_query)

    start_time = time.time()  # Start timing
    with st.spinner("In progress..."):  # Spinner to show processing
        with st.chat_message("AI"):
            # Get the response, generated queries, and retrieved documents
            response, queries, documents = get_response(user_query, st.session_state.chat_history, st.session_state.question_history)
            print("==========   PROCESS ENDED  ==========")
            # Calculate response time
            response_time = time.time() - start_time

            # Display the AI's response with the response time
            st.markdown(f"{response}\n\n**Response time:** {response_time:.1f}s")

    # Append the AI response to the session state chat history
    st.session_state.chat_history.append(AIMessage(content=response))
    st.session_state.question_history.append(HumanMessage(content=queries))

    # Clear the system cache after processing the response
    chromadb.api.client.SharedSystemClient.clear_system_cache()

    with st.sidebar:
        # Display the token count in the sidebar
        st.markdown(f"### Total Chat Token Count: {st.session_state.total_tokens}")
        st.markdown(f"### Total Chat Cost (USD): ${st.session_state.total_cost:.6f}")  # Display total cost

        # List the generated queries and retrieved documents in the sidebar
        st.markdown("### Similar questions:")
        for idx, query in enumerate(queries, start=1):
            st.write(f"{idx}. {query}")  
    
        st.markdown("### Retrieved documents:")
        st.write(documents)