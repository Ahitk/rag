import numpy as np
import os
from dotenv import load_dotenv
from langchain_openai.embeddings import OpenAIEmbeddings

# Load API Keys from environment variables
load_dotenv()  # Load environment variables from a .env file

# Retrieve API keys from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
embedding = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

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
            f"Cosine Similarity: {similarity * 100:.2f}%\n\n"
            f"Source file: {source}\n\n"
            f"Context:\n\n{content}\n"
        )

    return f"\n\n".join(formatted_docs)