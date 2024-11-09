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

# === Configuration Section ===

# Directory containing RAG data
data_directory = "/Users/taha/Desktop/rag/data"

# Load environment variables (e.g., API keys) from a .env file
load_dotenv()

# Retrieve API keys from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # OpenAI API Key
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")  # Tavily API Key

# Define maximum token limit for language models
MAX_TOKENS = 8192

# Initialize the GPT model and embeddings
model = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY, max_tokens=MAX_TOKENS)
embedding = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
#embedding = OpenAIEmbeddings(model="text-embedding-3-large", api_key=OPENAI_API_KEY)

# Available OpenAI models dictionary for selection
models = {
    "GPT-4o mini: Affordable and intelligent small model for fast, lightweight tasks": "gpt-4o-mini",
    "GPT-4o: High-intelligence flagship model for complex, multi-step tasks": "gpt-4o",
    "GPT-4: The previous set of high-intelligence model": "gpt-4",
    "GPT-3.5 Turbo: A fast, inexpensive model for simple tasks": "gpt-3.5-turbo-0125",
    "o1-preview: reasoning model designed to solve hard problems across domains": "o1-preview",
    "o1-mini: faster and cheaper reasoning model particularly good at coding, math, and science": "o1-mini",
}

# === Utility Functions ===

def prune_chat_history_if_needed():
    """
    Ensures the chat history stays within the token limit by pruning older entries.
    """
    total_token_count = sum([len(m.content.split()) for m in st.session_state.chat_history])
    while total_token_count > MAX_TOKENS:
        st.session_state.chat_history.pop(0)
        total_token_count = sum([len(m.content.split()) for m in st.session_state.chat_history])


def cosine_similarity(vec1, vec2):
    """
    Computes the cosine similarity between two vectors.
    Parameters:
        - vec1, vec2: Numpy arrays representing vector embeddings.
    Returns:
        - float: Cosine similarity value.
    """
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2) if (norm_vec1 and norm_vec2) else 0.0

# metadata not included
def format_documents(docs, question):
    """
    Formats retrieved documents, sorting by cosine similarity, and includes unique documents.
    Parameters:
        - docs: List of document objects.
        - question: Query question for context.
    Returns:
        - str: Formatted string of documents and similarity scores.
    """
    unique_docs = set()
    formatted_docs = []
    question_embedding = embedding.embed_query(question)
    doc_with_similarity = []

    for doc in docs:
        content_hash = hash(doc.page_content.strip())  # Doküman içeriğini benzersiz bir şekilde temsil eden hash
        if content_hash not in unique_docs:
            unique_docs.add(content_hash)  # Tekrar eden dokümanları engellemek için hash'i set'e ekle
            document_embedding = embedding.embed_query(doc.page_content)
            similarity = cosine_similarity(question_embedding, document_embedding)
            doc_with_similarity.append((doc, similarity))

    # Similarity'e göre sırala (azalan)
    doc_with_similarity.sort(key=lambda x: x[1], reverse=True)

    # Formatlama işlemi
    for i, (doc, similarity) in enumerate(doc_with_similarity, start=1):
        content = doc.page_content.strip() or "This document content is empty."
        formatted_docs.append(
            f"**{i}. Document:**\n\n"
            f"Cosine Similarity: {similarity * 100:.0f}%\n\n"
            f"Context:\n\n{content}\n"
        )

    return "\n\n".join(formatted_docs)


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
        #source = doc.metadata.get("source", "No source")
        content = doc.page_content

        # Append the formatted document string
        formatted_docs.append(
            f"**{i}. Document:** \n\n"  # Document title in bold with numbering
            f"Fusion Score: {score:.2f}\n\n"
            f"Cosine Similarity: {similarity * 100:.0f}%\n\n"
            #f"Source file: {source}\n\n"
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


# === Dense X summary Functions ===

def create_summary(doc_content):
    """
    Summarizes document content in German while preserving its semantic meaning.
    Parameters:
        - doc_content: Text content of the document.
    Returns:
        - str: Summarized text.
    """
    summary_template = ChatPromptTemplate.from_template(
        "Summarize the following document in German in a way that captures its semantic meaning most accurately, structured as minimal, self-contained units of meaning, and expressed in just one sentence.\n\n{doc}"
    )
    chain = summary_template | model | StrOutputParser()
    return chain.invoke({"doc": doc_content})


def summarize(data_directory):
    """
    Summarizes all text documents in the specified directory and stores the results in a file named '_summary.txt' 
    within each folder.

    Parameters:
        - data_directory (str): Path to the directory containing folders of text documents.
    """
    # Traverse the directory tree starting from the root directory.
    for root, dirs, files in os.walk(data_directory):
        # Check if the current level corresponds to the root of the specified directory.
        if root == data_directory:
            # Iterate over all subfolders in the root directory.
            for folder in dirs:
                # Define the path to the current folder.
                folder_path = os.path.join(data_directory, folder)
                
                # Define the path to the summary file to be created within this folder.
                summary_file_path = os.path.join(folder_path, '_summary.txt')
                
                # Skip the folder if the summary file already exists.
                if os.path.exists(summary_file_path):
                    continue
                
                # Open the summary file in write mode.
                with open(summary_file_path, 'w') as summary_file:
                    # Walk through the current folder and its subfolders.
                    for sub_root, _, sub_files in os.walk(folder_path):
                        # Process each file in the current folder/subfolder.
                        for file_name in sub_files:
                            # Exclude the summary file itself and ensure the file has a '.txt' extension.
                            if file_name != '_summary.txt' and file_name.endswith('.txt'):
                                # Construct the full file path.
                                file_path = os.path.join(sub_root, file_name)
                                
                                # Open and read the content of the text file.
                                with open(file_path, 'r', encoding='utf-8') as txt_file:
                                    content = txt_file.read()
                                
                                # Generate a summary of the file content using the `create_summary` function.
                                summary = create_summary(content)
                                
                                # Write the summary to the summary file, including the file path for reference.
                                summary_file.write(
                                    f"\n=== Chunk ===\n[File path: {file_path}\nFile summary: {summary}]\n"
                                )

import os

def summarize_with_filename(data_directory): 
    """
    Function to generate summary files (_summary.txt) for all .txt files 
    inside subdirectories of the given data directory.

    Args:
        data_directory (str): Path to the main data directory.

    Functionality:
    - Iterates through all subdirectories of the data directory.
    - Creates a '_summary.txt' file in each subdirectory.
    - If '_summary.txt' already exists, the subdirectory is skipped.
    - Reads the content of all .txt files (excluding '_summary.txt') in the subdirectory.
    - Summarizes the content using the `create_summary` function.
    - Writes the summary into '_summary.txt' along with the file names.
    """

    # Traverse the directory tree starting from data_directory
    for root, dirs, files in os.walk(data_directory):
        # Process only the top-level subdirectories
        if root == data_directory:
            for folder in dirs:
                # Construct the full path of the current subdirectory
                folder_path = os.path.join(data_directory, folder)
                
                # Define the path for the '_summary.txt' file in the subdirectory
                summary_file_path = os.path.join(folder_path, '_summary.txt')
                
                # Skip subdirectory if '_summary.txt' already exists
                if os.path.exists(summary_file_path):
                    print(f"{folder} folder already contains _summary.txt, skipping.")
                    continue
                
                # Create and open '_summary.txt' file for writing in the subdirectory
                with open(summary_file_path, 'w') as summary_file:
                    # Traverse the current subdirectory and its subdirectories
                    for sub_root, sub_dirs, sub_files in os.walk(folder_path):
                        for file_name in sub_files:
                            # Skip '_summary.txt' and non-txt files
                            if file_name != '_summary.txt' and file_name.endswith('.txt'):
                                # Construct the full path of the current .txt file
                                file_path = os.path.join(sub_root, file_name)
                                
                                # Read the content of the .txt file
                                with open(file_path, 'r', encoding='utf-8') as txt_file:
                                    content = txt_file.read()
                                
                                # Summarize the content using the `create_summary` function
                                summary = create_summary(content)
                                
                                # Write the file name and its summary to '_summary.txt'
                                summary_file.write(
                                    f"\n=== Chunk ===\n"
                                    f"[File name: {file_name}\n"
                                    f"File summary: {summary}]\n"
                                )
                
                # Indicate completion of '_summary.txt' creation for the current folder
                print(f"_summary.txt has been written for the folder: {folder}")