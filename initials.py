import os
import tiktoken
import numpy as np
from dotenv import load_dotenv
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.load import dumps, loads

# Load API Keys from environment variables
load_dotenv()  # Load environment variables from a .env file

# Retrieve API keys from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
embedding = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

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

# Function to format fusion_docs as a readable string with similarity scores
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

def format_docs_fusion(docs, question):
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
            doc_with_similarity.append((doc, score, similarity, source))

    # Sort documents by cosine similarity in descending order
    doc_with_similarity.sort(key=lambda x: x[1], reverse=True)

    # Format documents
    for i, (doc, similarity, source, score) in enumerate(doc_with_similarity, 1):
        # Use a placeholder message if the document content is empty
        content = doc.page_content.strip() or "This document content is empty."
        # Format the document's source, similarity score, and content
        formatted_docs.append(
            f"**{i}. Document:** \n\n"  # Document title in bold
            f"Cosine Similarity: {similarity * 100:.0f}%\n\n"
            f"Fusion Score: {score * 100:.0f}%\n\n"
            f"Source file: {source}\n\n"
            f"Context:\n\n{content}\n"
        )

    return f"\n\n".join(formatted_docs)

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

