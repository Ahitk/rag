import os
import tiktoken
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.load import dumps, loads
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Define the directory containing the rag data
data_directory = "/Users/taha/Desktop/rag/data"

# Load API Keys from environment variables
load_dotenv()  # Load environment variables from a .env file

# Retrieve API keys from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Token limit
MAX_TOKENS = 8192

model = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY, max_tokens=MAX_TOKENS)
embedding = OpenAIEmbeddings(model="text-embedding-3-large", api_key=OPENAI_API_KEY)

# Available OpenAI models
models = {
    "GPT-4o mini: Affordable and intelligent small model for fast, lightweight tasks": "gpt-4o-mini",
    "GPT-4o: High-intelligence flagship model for complex, multi-step tasks": "gpt-4o",
    "GPT-4: The previous set of high-intelligence model": "gpt-4",
    "GPT-3.5 Turbo: A fast, inexpensive model for simple tasks": "gpt-3.5-turbo-0125",
}

# Function to prune chat history to stay within token limit
def prune_chat_history_if_needed():
    total_token_count = sum([len(m.content.split()) for m in st.session_state.chat_history])
    while total_token_count > MAX_TOKENS:
        st.session_state.chat_history.pop(0)
        total_token_count = sum([len(m.content.split()) for m in st.session_state.chat_history])
  

# Calculate cosine similarity between two vectors
def cosine_similarity(vec1, vec2):
    """
    Computes the cosine similarity between two vectors.
    
    Parameters:
    - vec1 (np.ndarray): The first vector.
    - vec2 (np.ndarray): The second vector.
    
    Returns:
    - float: The cosine similarity between vec1 and vec2.
    """
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2) if (norm_vec1 and norm_vec2) else 0.0

## Multi-Query: Format streamlit output text
def format_docs(docs, question):
    """
    Formats the retrieved documents with their source and cosine similarity score, sorted by similarity.

    This function takes a list of documents and formats them to include the source of each document,
    its cosine similarity to the query embedding, and presents them in a numbered format.

    Args:
        docs (list): A list of documents retrieved from the database.
        question (str): The question to which the documents are relevant.

    Returns:
        str: A formatted string of documents including source, similarity score, and content.
    """
    # Initialize a set to track unique sources
    unique_sources = set()
    formatted_docs = []
    question_embedding = embedding.embed_query(question)

    # Collect documents with their similarity scores
    doc_with_similarity = []

    for doc in docs:
        # Retrieve the source of the document from its metadata, or set to "Web" if metadata is missing
        source = doc.metadata.get("source", "Web")  # "No source" durumunda "Web" olarak atanıyor
        
        # Check if the source is unique, add only unique sources
        if source not in unique_sources:
            unique_sources.add(source)
            # Compute the embedding of the document's content
            document_embedding = embedding.embed_query(doc.page_content)
            # Calculate cosine similarity between the query and document embeddings
            similarity = cosine_similarity(question_embedding, document_embedding)
            # Add document, similarity, source, and content to the list
            doc_with_similarity.append((doc, similarity, source))

    # Sort documents by cosine similarity in descending order
    doc_with_similarity.sort(key=lambda x: x[1], reverse=True)

    # Format documents
    for i, (doc, similarity, source) in enumerate(doc_with_similarity, start=1):
        # Use a placeholder message if the document content is empty
        content = doc.page_content.strip() or "This document content is empty."
        # Format the document's source, similarity score, and content
        formatted_docs.append(
            f"**{i}. Document:**\n\n"  # Document title in bold with numbering
            f"Cosine Similarity: {similarity * 100:.0f}%\n\n"
            f"Source file: {source}\n\n"
            f"Context:\n\n{content}\n"
        )

    return "\n\n".join(formatted_docs)

def format_docs_old(docs, question):
    """
    Formats the retrieved documents with their source and cosine similarity score, sorted by similarity.

    This function takes a list of documents and formats them to include the source of each document,
    its cosine similarity to the query embedding, and presents them in a numbered format.

    Args:
        docs (list): A list of documents retrieved from the database.

    Returns:
        list: A list of formatted strings containing the source, similarity score, and content of each document.
    """
    # Initialize a set to track unique sources
    unique_sources = set()
    formatted_docs = []
    question_embedding = embedding.embed_query(question)

    # Collect documents with their similarity scores
    doc_with_similarity = []

    for doc in docs:
        # Retrieve the source of the document from its metadata
        source = doc.metadata.get("source")
        
        # Check if the source is unique
        if source and source not in unique_sources:
            unique_sources.add(source)
            # Compute the embedding of the document's content
            document_embedding = embedding.embed_query(doc.page_content)
            # Calculate cosine similarity between the query and document embeddings
            similarity = cosine_similarity(question_embedding, document_embedding)
            # Add document, similarity, source, and content to the list
            doc_with_similarity.append((doc, similarity, source))

    # Sort documents by cosine similarity in descending order
    doc_with_similarity.sort(key=lambda x: x[1], reverse=True)

    # Format documents
    for i, (doc, similarity, source) in enumerate(doc_with_similarity, 1):
        # Use a placeholder message if the document content is empty
        content = doc.page_content.strip() or "This document content is empty."
        # Format the document's source, similarity score, and content
        formatted_docs.append(
            f"**{i}. Document:** \n\n"  # Document title in bold
            f"Cosine Similarity: {similarity * 100:.0f}%\n\n"
            f"Source file: {source}\n\n"
            f"Context:\n\n{content}\n"
        )

    return f"\n\n".join(formatted_docs)

# RAG-Fusion: Function to format fusion_docs as a readable string with similarity scores
def format_fusion_docs_with_similarity(fusion_docs, question):
    """
    Formats the fusion documents with their scores and cosine similarity to the question.
    
    Parameters:
    - fusion_docs (list[tuple]): A list of tuples containing documents and their scores.
    
    Returns:
    - str: A formatted string containing each document's source, fusion score, cosine similarity, and content.
    """
    # Sort the documents by their Fusion score in descending order
    fusion_docs = sorted(fusion_docs, key=lambda x: x[1], reverse=True)

    formatted_docs = []
    question_embedding = embedding.embed_query(question)
    
    for i, (doc, score) in enumerate(fusion_docs, start=1):
        doc_embedding = embedding.embed_query(doc.page_content)
        similarity = cosine_similarity(question_embedding, doc_embedding)
        source = doc.metadata.get("source", "No source")
        content = doc.page_content

        # Append the formatted document string
        formatted_docs.append(
            f"**{i}. Document:** \n\n"  # Document title in bold with numbering
            f"Fusion Score: {score:.2f}\n\n"
            f"Cosine Similarity: {similarity * 100:.0f}%\n\n"
            f"Source file: {source}\n\n"
            f"Context:\n\n{content}\n"
        )

    return "\n".join(formatted_docs)


# Multi-Query: Retrieve and return unique documents
def get_unique_union(documents):
    """
    Returns a unique union of retrieved documents by flattening and removing duplicates.
    """
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    unique_docs = list(set(flattened_docs))  # Remove duplicates
    return [loads(doc) for doc in unique_docs]

# RAG-Fusion: Function for Reciprocal Rank Fusion (RRF)
def reciprocal_rank_fusion(results: list[list], k=60):
    """
    Applies Reciprocal Rank Fusion (RRF) to combine multiple lists of ranked documents.
    
    Parameters:
    - results (list[list]): A list of lists where each inner list contains ranked documents.
    - k (int): An optional parameter for the RRF formula, default is 60.
    
    Returns:
    - list: A list of tuples where each tuple contains a document and its fused score.
    """
    
    # Initialize a dictionary to store the fused scores for each unique document
    fused_scores = {}

    # Iterate through each list of ranked documents
    for docs in results:
        # Iterate through each document in the list, with its rank (position in the list)
        for rank, doc in enumerate(docs):
            # Serialize the document to a string format to use as a key
            doc_str = dumps(doc)
            # Initialize the document's score if not already present
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            # Update the document's score using the RRF formula: 1 / (rank + k)
            fused_scores[doc_str] += 1 / (rank + k)

    # Sort documents based on their fused scores in descending order
    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]

    # Return the reranked results as a list of tuples
    return reranked_results


# Tokenizer
def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def get_token_count(docs, question, prompt, chat_history):
    """
    Calculate and return token counts for the prompt, question, retrieved documents, and total.
    """
    # Calculate token counts for different components
    prompt_tokens = num_tokens_from_string(prompt.format(context="dummy", question=question, chat_history=chat_history), "cl100k_base")
    question_tokens = num_tokens_from_string(question, "cl100k_base")
    docs_tokens = sum([num_tokens_from_string(doc.page_content, "cl100k_base") for doc in docs])
    
    # Total token count including prompt, question, and documents
    total_tokens = prompt_tokens + question_tokens + docs_tokens
    
    return question_tokens, docs_tokens, prompt_tokens, total_tokens

# Özetleme zincirini oluşturma
def create_summary(doc_content):
    summary_template = ChatPromptTemplate.from_template(
        "Summarize the following document in German in a way that captures its semantic meaning most accurately.\n\n{doc}"
    )
    chain = summary_template | model | StrOutputParser()
    return chain.invoke({"doc": doc_content})

# Ana klasörün yolu (data klasörünün yolu)

def summarize(data_directory): 
    # data klasörü altındaki her bir ana klasör için işlem yapıyoruz
    for root, dirs, files in os.walk(data_directory):
        print("aaaaaa")
        if root == data_directory:
            for folder in dirs:
                folder_path = os.path.join(data_directory, folder)
                
                # _summary.txt dosyasının yolu
                summary_file_path = os.path.join(folder_path, '_summary.txt')
                
                # Eğer _summary.txt zaten varsa bu klasörü atla
                if os.path.exists(summary_file_path):
                    print(f"{folder} klasöründe _summary.txt dosyası zaten mevcut, atlanıyor.")
                    continue
                
                # _summary.txt dosyasını bu ana klasör içinde oluşturuyoruz (alfabetik olarak en üstte olacak şekilde)
                with open(summary_file_path, 'w') as summary_file:
                    # Bu klasörün altındaki tüm dosyaları listelemek için tekrar os.walk kullanıyoruz
                    for sub_root, sub_dirs, sub_files in os.walk(folder_path):
                        for file_name in sub_files:
                            if file_name != '_summary.txt' and file_name.endswith('.txt'):
                                # Her txt dosyasının tam yolunu alıyoruz
                                file_path = os.path.join(sub_root, file_name)
                                
                                # Dosyanın içeriğini okuyoruz
                                with open(file_path, 'r', encoding='utf-8') as txt_file:
                                    content = txt_file.read()
                                
                                # Belgeyi özetliyoruz
                                summary = create_summary(content)
                                
                                # Dosya yolunu ve özetini _summary.txt dosyasına yazıyoruz
                                summary_file.write(f"\n=== Chunk ===\n[File path: {file_path}\nFile summary: {summary}]\n")
                
                print(f"{folder} klasörüne _summary.txt dosyası yazıldı.")

                import os




def summarize_with_filename(data_directory): 
    # data klasörü altındaki her bir ana klasör için işlem yapıyoruz
    for root, dirs, files in os.walk(data_directory):
        if root == data_directory:
            for folder in dirs:
                folder_path = os.path.join(data_directory, folder)
                
                # _summary.txt dosyasının yolu
                summary_file_path = os.path.join(folder_path, '_summary.txt')
                
                # Eğer _summary.txt zaten varsa bu klasörü atla
                if os.path.exists(summary_file_path):
                    print(f"{folder} klasöründe _summary.txt dosyası zaten mevcut, atlanıyor.")
                    continue
                
                # _summary.txt dosyasını bu ana klasör içinde oluşturuyoruz
                with open(summary_file_path, 'w') as summary_file:
                    # Bu klasörün altındaki tüm dosyaları listelemek için tekrar os.walk kullanıyoruz
                    for sub_root, sub_dirs, sub_files in os.walk(folder_path):
                        for file_name in sub_files:
                            if file_name != '_summary.txt' and file_name.endswith('.txt'):
                                # Her txt dosyasının tam yolunu alıyoruz
                                file_path = os.path.join(sub_root, file_name)
                                
                                # Dosyanın içeriğini okuyoruz
                                with open(file_path, 'r', encoding='utf-8') as txt_file:
                                    content = txt_file.read()
                                
                                # Belgeyi özetliyoruz
                                summary = create_summary(content)
                                
                                # Dosya adını ve özetini _summary.txt dosyasına yazıyoruz
                                summary_file.write(f"\n=== Chunk ===\n[File name: {file_name}\nFile summary: {summary}]\n")
                
                print(f"{folder} klasörüne _summary.txt dosyası yazıldı.")