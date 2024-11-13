import gc  
import glob  
import os  
import random  
from langchain.docstore.document import Document  
from langchain_experimental.text_splitter import SemanticChunker  # Advanced semantic text splitting
from langchain_text_splitters import CharacterTextSplitter  # Basic text splitting by character
from langchain.retrievers import EnsembleRetriever  # Hybrid: Combine multiple retrieval methods
from langchain_community.retrievers import BM25Retriever  # BM25-based keyword search
from langchain_community.document_loaders import DirectoryLoader, TextLoader  
from langchain_chroma import Chroma  
from routing import get_specific_directory  


# ============================= CONSTANTS =============================
# These constants control parameters like number of retrieved documents
TOP_N = 30  # Routing: Number of closest documents to retrieve
HYBRID_VECTORSTORE_WEIGHT = 0.5  # Weight for combining semantic and keyword search
MAX_CHUNK_NUMBER = 5  # Maximum chunks allowed per document
MAX_DOCUMENT_NUMBER_K = 10  # Maximum documents for retrieval
SUMMARY_FILE_PATTERN = '**/_summary.txt'  # Pattern for summary files

# Global variables for vectorstore and retriever
vectorstore = None
retriever = None

# Trigger garbage collection to optimize memory
gc.collect()

# ============================== DENSE X SUMMARY-BASED FILE EXTRACTION ==============================

def extract_summaries_from_files(summary_dir):
    """
    Extract summaries from `_summary.txt` files in the given directory.

    Args:
        summary_dir (str): Directory containing `_summary.txt` files.

    Returns:
        dict: A dictionary mapping file names to their respective summaries.
    """
    # Find all `_summary.txt` files in the specified directory
    summary_file_paths = glob.glob(os.path.join(summary_dir, SUMMARY_FILE_PATTERN), recursive=True)
    file_summary_map = {}

    # Parse each summary file
    for summary_file in summary_file_paths:
        with open(summary_file, 'r') as f:
            content = f.read()

        # Split content into chunks using the delimiter
        chunks = content.split("=== Chunk ===")
        for chunk in chunks:
            if "File name:" in chunk and "File summary:" in chunk:
                try:
                    # Extract file name and summary
                    file_name_line = next(line for line in chunk.split('\n') if "File name:" in line)
                    summary_line = next(line for line in chunk.split('\n') if "File summary:" in line)
                    
                    file_name = file_name_line.split("File name:")[1].strip()
                    summary_text = summary_line.split("File summary:")[1].strip()
                    file_summary_map[file_name] = summary_text
                except StopIteration:
                    print(f"Warning: Skipping chunk due to formatting issues in file: {summary_file}")

    return file_summary_map

# ============================== INITIALIZING SUMMARY VECTORSTORE ==============================

def initialize_summary_vectorstore(file_summaries, embedding_model):
    """
    Create a vectorstore using summaries.

    Args:
        file_summaries (dict): Mapping of file names to summaries.
        embedding_model: Model used to embed summaries.

    Returns:
        Chroma: A vectorstore initialized with embedded summaries.
    """
    # Convert summaries into LangChain Document objects
    summary_documents = [Document(page_content=summary) for summary in file_summaries.values()]
    # Create a Chroma vectorstore with the documents
    summary_vector_store = Chroma.from_documents(documents=summary_documents, embedding=embedding_model)
    return summary_vector_store

# ============================== SUMMARY RETRIEVAL ==============================

def retrieve_closest_filenames(user_question, summary_retriever, summary_map, top_n=TOP_N):
    """
    Retrieve the closest filenames based on the user's query.

    Args:
        user_question (str): The user's input query.
        summary_retriever: Retriever object for querying summaries.
        summary_map (dict): Mapping of filenames to summaries.
        top_n (int): Number of closest results to return.

    Returns:
        list: List of filenames corresponding to the retrieved summaries.
    """
    retrieved_filenames = []
    seen_files = set()
    retries = 0

    # Continue retrieving until the desired number of unique filenames is reached
    while len(retrieved_filenames) < top_n and retries < 5:
        results = summary_retriever.invoke(user_question)
        if results is None:
            print(f"No results retrieved for question: {user_question}")
            break

        for result in results:
            summary_content = result.page_content
            file_name = next((name for name, summary in summary_map.items() if summary == summary_content), None)
            
            # Ensure uniqueness of retrieved filenames
            if file_name and file_name not in seen_files:
                retrieved_filenames.append(file_name)
                seen_files.add(file_name)
            
            if len(retrieved_filenames) >= top_n:
                break

        retries += 1

    print(f"========== NUMBER OF DOCUMENTS RETRIEVED: {len(retrieved_filenames)} ==========")
    if len(retrieved_filenames) < top_n:
        print(f"Warning: Only {len(retrieved_filenames)} unique results were found after {retries} iterations.")

    return retrieved_filenames

# ============================== LOADING ORIGINAL DOCUMENTS ==============================

def load_original_docs_by_filenames(file_names, doc_directory):
    """
    Load original documents based on filenames.

    Args:
        file_names (list): List of filenames to load.
        doc_directory (str): Directory containing the original documents.

    Returns:
        list: A list of Document objects representing the loaded files.
    """
    loaded_documents = []
    for file_name in file_names:
        # Find the file path in the directory
        file_path = next((os.path.join(root, file_name) for root, _, files in os.walk(doc_directory) if file_name in files), None)
        
        if file_path and os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                loaded_documents.append(Document(page_content=content))
            except Exception as e:
                print(f"Error loading document from {file_name}: {e}")
        else:
            print(f"Original document not found for file: {file_name}")

    return loaded_documents

# ============================== GENERATE VECTORSTORE ==============================
# without routing
def generate_final_vectorstore_with_chunks(user_question, doc_directory, embedding_model):
    """
    Create a vectorstore with semantic chunking applied to documents.

    Args:
        user_question (str): User's query to retrieve relevant documents.
        doc_directory (str): Directory containing the documents.
        embedding_model: Model used for embedding.

    Returns:
        Chroma: Vectorstore with embedded chunks.
    """
    # Step 1: Load summaries and create a vectorstore
    file_summaries = extract_summaries_from_files(doc_directory)
    summary_vectorstore = initialize_summary_vectorstore(file_summaries, embedding_model)
    
    # Step 2: Retrieve relevant summaries
    summary_retriever = summary_vectorstore.as_retriever(search_kwargs={"k": TOP_N})
    closest_file_names = retrieve_closest_filenames(user_question, summary_retriever, file_summaries, top_n=TOP_N)
    
    # Clear vectorstore memory
    summary_vectorstore.delete_collection()

    # Step 3: Load original documents
    original_documents = load_original_docs_by_filenames(closest_file_names, doc_directory)
    print(f"========== {len(original_documents)} DOCUMENTS SUCCESSFULLY LOADED ==========")

    # Step 4: Apply semantic chunking
    print("========== SEMANTIC CHUNKING IN PROGRESS ==========")
    semantic_splitter = SemanticChunker(embedding_model)
    document_chunks = []
    for doc in original_documents:
        split_chunks = semantic_splitter.create_documents([doc.page_content])
        document_chunks.extend(split_chunks)
    print(f"========== TOTAL CHUNKS CREATED: {len(document_chunks)} ==========")

    # Step 5: Create a final vectorstore from the chunks
    final_chunked_vectorstore = Chroma.from_documents(documents=document_chunks, embedding=embedding_model)
    print("========== FINAL VECTOR STORE WITH CHUNKS CREATED ==========")
    
    return final_chunked_vectorstore


def generate_vectorstore_semantic_chunking(user_question, model, doc_directory, embedding_model):
    """
    Create a vectorstore with semantic chunking applied to documents.

    Args:
        user_question (str): User's query to retrieve relevant documents.
        doc_directory (str): Directory containing the documents.
        embedding_model: Model used for embedding.

    Returns:
        Chroma: Vectorstore with embedded chunks.
    """
    # Step 1: Load summaries and create a vectorstore
    file_summaries = extract_summaries_from_files(get_specific_directory(user_question, model, doc_directory))
    summary_vectorstore = initialize_summary_vectorstore(file_summaries, embedding_model)
    
    # Step 2: Retrieve relevant summaries
    summary_retriever = summary_vectorstore.as_retriever(search_kwargs={"k": TOP_N})
    closest_file_names = retrieve_closest_filenames(user_question, summary_retriever, file_summaries, top_n=TOP_N)
    
    # Clear vectorstore memory
    summary_vectorstore.delete_collection()

    # Step 3: Load original documents
    original_documents = load_original_docs_by_filenames(closest_file_names, doc_directory)
    print(f"========== {len(original_documents)} DOCUMENTS SUCCESSFULLY LOADED ==========")

    # Step 4: Apply semantic chunking
    print("========== SEMANTIC CHUNKING IN PROGRESS ==========")
    semantic_splitter = SemanticChunker(embedding_model)
    document_chunks = []
    for doc in original_documents:
        split_chunks = semantic_splitter.create_documents([doc.page_content])
        document_chunks.extend(split_chunks)
    print(f"========== TOTAL CHUNKS CREATED: {len(document_chunks)} ==========")
    
    # Step 5: Create a final vectorstore from the chunks
    final_chunked_vectorstore = Chroma.from_documents(documents=document_chunks, embedding=embedding_model)
    print("========== FINAL VECTOR STORE WITH CHUNKS CREATED ==========")
    
    return final_chunked_vectorstore

# without chunking, documents direct loaded to vector store
def generate_vectorstore(user_question, model, doc_directory, embedding_model):
    """
    Create a vectorstore with semantic chunking applied to documents.

    Args:
        user_question (str): User's query to retrieve relevant documents.
        doc_directory (str): Directory containing the documents.
        embedding_model: Model used for embedding.

    Returns:
        Chroma: Vectorstore with embedded chunks.
    """
    # Step 1: Load summaries and create a vectorstore
    file_summaries = extract_summaries_from_files(get_specific_directory(user_question, model, doc_directory))
    summary_vectorstore = initialize_summary_vectorstore(file_summaries, embedding_model)
    
    # Step 2: Retrieve relevant summaries
    summary_retriever = summary_vectorstore.as_retriever(search_kwargs={"k": TOP_N})
    closest_file_names = retrieve_closest_filenames(user_question, summary_retriever, file_summaries, top_n=TOP_N)
    
    # Clear vectorstore memory
    summary_vectorstore.delete_collection()

    # Step 3: Load original documents
    original_documents = load_original_docs_by_filenames(closest_file_names, doc_directory)
    print(f"========== {len(original_documents)} DOCUMENTS SUCCESSFULLY LOADED ==========")

    final_chunked_vectorstore = Chroma.from_documents(documents=original_documents, embedding=embedding_model)
    print("========== FINAL VECTOR STORE WITH CHUNKS CREATED ==========")
    
    return final_chunked_vectorstore

# ============================== CHARACTER SPLITTING ==============================

def generate_final_vectorstore_with_character_splitter(user_question, doc_directory, embedding_model):
    """
    Create a vectorstore by applying character-based splitting to documents.

    Args:
        user_question (str): Query for retrieving documents.
        doc_directory (str): Directory containing documents.
        embedding_model: Model used for embedding.

    Returns:
        Chroma: Vectorstore with character-split chunks.
    """
    # Load summaries from the directory
    file_summaries = extract_summaries_from_files(doc_directory)
    summary_vectorstore = initialize_summary_vectorstore(file_summaries, embedding_model)

    # Retrieve relevant summaries and filenames
    summary_retriever = summary_vectorstore.as_retriever(search_kwargs={"k": TOP_N})
    closest_file_names = retrieve_closest_filenames(user_question, summary_retriever, file_summaries, top_n=TOP_N)
    
    # Clear the vectorstore memory
    summary_vectorstore.delete_collection()

    # Load original documents
    original_documents = load_original_docs_by_filenames(closest_file_names, doc_directory)
    print(f"========== {len(original_documents)} DOCUMENTS SUCCESSFULLY LOADED ==========")

    # Character-based splitting
    print("========== CHARACTER SPLITTING IN PROGRESS ==========")
    text_splitter = CharacterTextSplitter(
        separator='',  # No specific separator
        chunk_size=500,  # Maximum size of each chunk
        chunk_overlap=50  # Overlap size to ensure continuity between chunks
    )

    document_chunks = []
    for doc in original_documents:
        split_chunks = text_splitter.create_documents([doc.page_content])
        document_chunks.extend(split_chunks)
    print(f"========== TOTAL CHUNKS CREATED: {len(document_chunks)} ==========")

    # Create a vectorstore from the character-based chunks
    final_chunked_vectorstore = Chroma.from_documents(documents=document_chunks, embedding=embedding_model)
    print("========== FINAL VECTOR STORE WITH CHUNKS CREATED ==========")
    
    return final_chunked_vectorstore


# ============================== DENSE X: SUMMARY BY FILE PATH ==============================

def load_summaries(data_directory):
    """
    Extract summaries and their associated file paths.

    Args:
        data_directory (str): Directory containing summary files.

    Returns:
        tuple: (summaries dict, data directory)
    """
    # Collect all summary files matching the pattern
    summary_files = glob.glob(os.path.join(data_directory, SUMMARY_FILE_PATTERN), recursive=True)
    summaries = {}

    # Process each summary file
    for file in summary_files:
        with open(file, 'r') as f:
            content = f.read()

        # Split the content into chunks
        chunks = content.split("=== Chunk ===")
        for chunk in chunks:
            # Identify valid chunks with "File path:" and "File summary:"
            if "File path:" in chunk and "File summary:" in chunk:
                try:
                    # Extract file path and summary information
                    lines = chunk.split('\n')
                    file_path_line = [line for line in lines if "File path:" in line]
                    summary_line = [line for line in lines if "File summary:" in line]

                    if file_path_line and summary_line:
                        file_path = file_path_line[0].split("File path:")[1].strip()
                        summary_text = summary_line[0].split("File summary:")[1].strip()
                        summaries[file_path] = summary_text
                except IndexError:
                    # Warn about potential formatting issues
                    print(f"Warning: Skipping chunk due to formatting issues in file: {file}")

    return summaries, data_directory

def create_chroma_vectorstore(summaries, embedding):
    """
    Create a Chroma vectorstore from the provided summaries.

    Args:
        summaries (dict): A dictionary mapping file paths to their summaries.
        embedding (Embedding): An embedding object to create vector representations.

    Returns:
        Chroma: A vectorstore containing the embedded summaries.
    """
    documents = []
    summaries_text = list(summaries.values())
    file_paths = list(summaries.keys())
    
    # Embed summaries using the provided embedding object
    summary_embeddings = embedding.embed_documents(summaries_text)
    
    # Create Document objects for each summary
    for i, summary in enumerate(summaries_text):
        doc = Document(page_content=summary, metadata={'source': file_paths[i]})
        documents.append(doc)
    
    # Create Chroma vectorstore from documents
    summary_vectorstore = Chroma.from_documents(documents=documents, embedding=embedding)
    return summary_vectorstore

def find_closest_summaries_with_chroma(question, summary_retriever, top_n=TOP_N):
    """
    Find the closest summaries to a question using Chroma retriever.

    Args:
        question (str): The user's query or question.
        summary_retriever (Retriever): A retriever object to query summaries.
        top_n (int): Number of top summaries to retrieve.

    Returns:
        list: Unique file paths of the closest summaries.
    """
    unique_paths = []
    seen_files = set()
    retries = 0  # To prevent infinite loops in case something goes wrong

    while len(unique_paths) < top_n and retries < 5:  # Limit retries to avoid infinite loops
        results = summary_retriever.invoke(question)

        for result in results:
            file_path = result.metadata['source']
            
            # Add unique file paths only
            if file_path not in seen_files:
                unique_paths.append(file_path)
                seen_files.add(file_path)
            
            # Stop once desired number of unique paths is reached
            if len(unique_paths) >= top_n:
                break
        
        retries += 1

    # Warn if fewer results than requested are found
    if len(unique_paths) < top_n:
        print(f"Warning: Only {len(unique_paths)} unique results were found after {retries} iterations.")

    return unique_paths

def load_original_documents_from_summary_paths(summary_paths):
    """
    Load the original documents corresponding to the provided summary paths.

    Args:
        summary_paths (list): List of paths to summary files.

    Returns:
        list: List of Document objects containing the original content.
    """
    docs = []
    for summary_path in summary_paths:
        if not os.path.exists(summary_path):
            print(f"Original document not found for summary: {summary_path}")
            continue
        
        try:
            with open(summary_path, 'r') as f:
                content = f.read()
            docs.append(Document(page_content=content, metadata={'source': summary_path}))
        except FileNotFoundError:
            print(f"Original document not found for summary: {summary_path}")
        except Exception as e:
            print(f"Error loading document from {summary_path}: {e}")
    
    return docs
 
# IMPORTANT: SEMANTIC CHUNKING IMPLEMENTATION
def get_vectorstore(question, model, data_directory, embedding):
    """
    Creates a vector store from documents relevant to the given question by performing semantic chunking.
    
    Parameters:
    - question (str): The input query for retrieving relevant documents.
    - model: The model used for determining the specific directory for summaries.
    - data_directory (str): The root directory containing the data files.
    - embedding: The embedding model used for vector representation.

    Returns:
    - vectorstore: A Chroma vector store created from semantically chunked documents.
    """

    # Step 1: Load summaries based on the question and model
    summaries, category = load_summaries(get_specific_directory(question, model, data_directory))
    
    # Step 2: Create a Chroma vector store from the summaries
    summary_vectorstore = create_chroma_vectorstore(summaries, embedding)
    
    # Step 3: Create a retriever from the vector store
    summary_retriever = summary_vectorstore.as_retriever(search_kwargs={"k": TOP_N})    
    
    # Step 4: Retrieve the closest summaries using Chroma
    closest_summary_files = find_closest_summaries_with_chroma(question, summary_retriever, top_n=TOP_N)
    
    # Step 5: Clean up the Chroma vector store to free up memory
    summary_vectorstore.delete_collection()  # WARNING: This deletes all vectors in the store
    
    # Step 6: Load the original documents corresponding to the retrieved summaries
    docs = load_original_documents_from_summary_paths(closest_summary_files)
    print(f"==========   {len(docs)} DOCUMENTS SUCCESSFULLY LOADED FROM DATA  ==========")

    # Informational log for semantic chunking process
    print("==========   SEMANTIC CHUNKING WORKING  ==========")
    
    # Step 7: Semantic Chunking using a custom text splitter
    text_splitter = SemanticChunker(embedding)  # Initialize the semantic chunker with the embedding model
    
    # Step 8: Split each document into smaller, meaningful chunks
    chunks = []
    for doc in docs:
        # Break document content into smaller semantic chunks
        split = text_splitter.create_documents([doc.page_content])  
        
        # Preserve original metadata for each chunk
        for chunk in split:
            chunk.metadata = doc.metadata  # Ensure metadata integrity
        
        # Extend the chunks list with the new chunks
        chunks.extend(split)
    print(f"==========   CHUNKS CREATED: {len(chunks)}  ==========")

    # Step 9: Create a new Chroma vector store from the chunks
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embedding)
    print("==========   VECTORSTORE CREATED  ==========")

    # Return the final vector store
    return vectorstore  # Optionally, you can also return the category if needed

###### FUNCTION: get_vectorstore_without_chunking ######
# This function processes a given question and dataset to create a vector store 
# (without semantic chunking) based on the provided model and embedding.
# It first retrieves summaries relevant to the question, clears temporary vector stores, 
# and then builds a vector store using the original documents.

def get_vectorstore_without_chunking(question, model, data_directory, embedding):
    """
    Generates a vector store without semantic chunking for a given question and dataset.

    Args:
        question (str): The question or query for which the vector store is created.
        model (str): The model used for processing the data.
        data_directory (str): Path to the directory containing data files.
        embedding (object): Embedding model used for vectorization.

    Returns:
        vectorstore (object): A vector store containing document embeddings.
    """
    # Load summaries from a specific directory based on the question and model
    summaries = load_summaries(get_specific_directory(question, model, data_directory))

    # Create a Chroma vector store using the summaries and embedding model
    summary_vectorstore = create_chroma_vectorstore(summaries, embedding)

    # Generate a retriever from the Chroma vector store
    # This retriever is used to search for the top-N relevant summaries
    summary_retriever = summary_vectorstore.as_retriever(search_kwargs={"k": TOP_N})

    # Identify the top-N closest summaries to the question
    closest_summary_files = find_closest_summaries_with_chroma(
        question, summary_retriever, top_n=TOP_N
    )

    # Clear the Chroma vector store after retrieving relevant summaries
    summary_vectorstore.delete_collection()  # Deletes all vectors in the collection
    # print("Summary vectorstore has been cleared.")  # Optional: Debug print statement

    # Load the original documents referred to by the closest summaries
    docs = load_original_documents_from_summary_paths(closest_summary_files)
    print(f"==========   {len(docs)} DOCUMENTS SUCCESSFULLY LOADED FROM DATA  ==========")

    # Create a vector store using the original documents and the embedding model
    vectorstore = Chroma.from_documents(documents=docs, embedding=embedding)
    print("==========   VECTORSTORE CREATED  ==========")

    return vectorstore

#============================== FOR TEST PURPOSES ONLY ================================

# Function to create a vector store using semantic chunking on text files in a given directory.
def get_vectorstore_semantic_chunking_no_summary(test_directory, embedding):
    """
    Generates a vector store by performing semantic chunking on text files in the specified directory.

    Parameters:
        test_directory (str): Path to the directory containing text files to be processed.
        embedding (object): Embedding model or object to generate vector representations.

    Returns:
        vectorstore (object): A vector store object created from the semantic chunks of the text files.
    """

    # Fetch all .txt files from the provided directory
    all_txt_files = glob.glob(os.path.join(test_directory, "*.txt"))

    # Initialize a list to store the content of all text files
    all_texts = []
    
    # Read and combine the content of all text files
    for file_path in all_txt_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            all_texts.append(f.read())  # Read the file content and add it to the list

    print("==========   SEMANTIC CHUNKING WORKING  ==========")

    # Initialize the semantic chunker with the provided embedding
    text_splitter = SemanticChunker(embedding)
    
    # Generate semantic chunks from the combined text content
    chunks = text_splitter.create_documents(all_texts)

    print(f"==========   CHUNKS CREATED: {len(chunks)}  ==========")

    # Create a vector store from the generated chunks using the embedding
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embedding)

    print("==========   VECTORSTORE CREATED  ==========")
    
    # Return the created vector store
    return vectorstore

def get_hybrid_semantic_retriever(question, model, data_directory, embedding):
    """
    Generates a hybrid semantic retriever by combining semantic and keyword-based search mechanisms.
    
    Parameters:
        question (str): The input query for which relevant documents need to be retrieved.
        model (Any): The model used for embedding or other related tasks.
        data_directory (str): Directory containing the data and summaries.
        embedding (Any): The embedding model for vectorizing text data.
    
    Returns:
        hybrid_retriever (EnsembleRetriever): A hybrid retriever combining semantic and keyword-based methods.
    """

    # === Step 1: Load Summaries ===
    # Load summaries relevant to the input question using a helper function.
    summaries = load_summaries(get_specific_directory(question, model, data_directory))
    
    # === Step 2: Create Chroma Vectorstore for Summaries ===
    # Initialize Chroma vector store with summaries and the provided embedding model.
    summary_vectorstore = create_chroma_vectorstore(summaries, embedding)
    
    # === Step 3: Find Closest Summaries ===
    # Use Chroma retriever to find the top-N most relevant summaries.
    summary_retriever = summary_vectorstore.as_retriever(search_kwargs={"k": TOP_N})
    closest_summary_files = find_closest_summaries_with_chroma(question, summary_retriever, top_n=TOP_N)
    
    # === Step 4: Clean Up Temporary Vectorstore ===
    # Delete the Chroma collection to free memory (clears all vectors stored).
    summary_vectorstore.delete_collection()  # This will delete all stored vectors.

    # === Step 5: Load Original Documents ===
    # Retrieve original documents using the paths of the closest summaries.
    docs = load_original_documents_from_summary_paths(closest_summary_files)
    print(f"==========   DOCUMENTS SUCCESSFULLY LOADED FROM DATA  ==========")

    # === Step 6: Semantic Chunking ===
    print("==========   SEMANTIC CHUNKING WORKING  ==========")
    # Initialize a semantic chunker to divide documents into smaller, meaningful chunks.
    text_splitter = SemanticChunker(embedding, number_of_chunks=MAX_CHUNK_NUMBER)
    
    # Process each document to create smaller, semantically meaningful chunks.
    chunks = []  # Initialize an empty list to store chunks.
    for doc in docs:
        # Split the document content into smaller chunks.
        split = text_splitter.create_documents([doc.page_content])
        
        # Preserve the original metadata for each chunk.
        for chunk in split:
            chunk.metadata = doc.metadata
        chunks.extend(split)  # Add the processed chunks to the list.
    
    print(f"==========   CHUNKS CREATED: {len(chunks)}  ==========")

    # === Step 7: Create a Vectorstore from Chunks ===
    # Build a new Chroma vector store using the document chunks.
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embedding)
    print("==========   VECTORSTORE CREATED  ==========")

    # === Step 8: Initialize Hybrid Retriever ===
    # Create semantic and keyword-based retrievers.
    semantic_retriever = vectorstore.as_retriever()
    keyword_retriever = BM25Retriever.from_documents(chunks)
    
    # Combine the retrievers using the EnsembleRetriever with specified weights.
    hybrid_retriever = EnsembleRetriever(
        retrievers=[keyword_retriever, semantic_retriever],
        weights=[1-HYBRID_VECTORSTORE_WEIGHT, HYBRID_VECTORSTORE_WEIGHT]
    )
    
    print("==========   HYBRID SEARCH FINISHED  ==========")
    
    # Return the hybrid retriever for further use.
    return hybrid_retriever

# Naive RAG (Retrieve and Generate) - Semantic Search - Character Splitting
# Function to create a vector store from text documents using a naive approach.
def get_vectorstore_naive_semantic(test_directory, embedding):
    """
    This function processes text files from a specified directory, splits their content into chunks,
    and generates a vector store for semantic search. It implements a naive retrieval mechanism 
    and supports character-based chunking.

    Args:
        test_directory (str): Path to the directory containing text files to be processed.
        embedding (object): Embedding model or function to encode text into vectors.

    Returns:
        tuple: A tuple containing:
            - vectorstore: The vector store created from the document chunks.
            - category (str): The name of the directory (used as a category label).
    """
    
    # Step 1: Gather all .txt files from the specified directory
    all_txt_files = glob(os.path.join(test_directory, "*.txt"))
    
    # Step 2: If there are more than 50 files, randomly select 50 of them
    selected_files = random.sample(all_txt_files, min(50, len(all_txt_files)))
    
    # Step 3: Read the contents of the selected files and store them in a list
    all_texts = []
    for file_path in selected_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            all_texts.append(f.read())
    
    # Step 4: Split the text contents into smaller chunks using CharacterTextSplitter
    # This helps in creating manageable document pieces for semantic search.
    text_splitter = CharacterTextSplitter(
        separator="\n\n",      # Define how to separate chunks (double newline in this case)
        chunk_size=200,        # Maximum size of each chunk
        chunk_overlap=30,      # Overlap between consecutive chunks for context preservation
        length_function=len,   # Function to measure chunk length
        is_separator_regex=False  # Indicates if the separator is a regex pattern
    )
    
    # Generate document chunks from the collected texts
    chunks = text_splitter.create_documents(all_texts)
    
    # Step 5: Create a vector store from the chunks using the provided embedding model
    # Chroma is used as the backend for the vector store.
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embedding)
    print("==========   VECTORSTORE CREATED  ==========")
    
    # Step 6: Determine the category name from the directory's name
    # Example: If test_directory is "path/to/category_name", category will be "category_name"
    category = os.path.basename(test_directory.rstrip('/'))
    
    # Step 7: Return the vector store and the category
    return vectorstore, category


# Advanced RAG - Semantic Search - Dense X - Semantic Splitting - Fusion - Logical Routing
# NOTE: This implementation directly uses the specified directory without directory routing.

def get_vectorstore_semantic(question, test_directory, embedding):
    """
    Processes a question to retrieve semantically relevant documents by leveraging vector stores,
    semantic chunking, and embedding-based search.
    
    Parameters:
        question (str): The query/question for semantic search.
        test_directory (str): Path to the directory containing summaries and documents.
        embedding (object): The embedding model used for vectorization.
        
    Returns:
        tuple: A vector store of semantically chunked documents and their category.
    """

    # === Load Summaries and Categories ===
    summaries, category = load_summaries(test_directory)  # Load summaries and their associated category.
    
    # === Create Chroma Vector Store from Summaries ===
    summary_vectorstore = create_chroma_vectorstore(summaries, embedding)  # Initialize vector store for summaries.

    # === Create Retriever for Semantic Search ===
    summary_retriever = summary_vectorstore.as_retriever(search_kwargs={"k": TOP_N})  # Create a retriever with 'k' nearest neighbors.

    # === Find Closest Summaries ===
    closest_summary_files = find_closest_summaries_with_chroma(question, summary_retriever, top_n=TOP_N)  
    # Retrieve top-N closest summaries to the given question.

    # === Cleanup: Delete the Chroma Vector Store ===
    summary_vectorstore.delete_collection()  # Clears the vector store to release memory.

    # === Load Original Documents Referenced by Summaries ===
    docs = load_original_documents_from_summary_paths(closest_summary_files)  
    print(f"==========   {len(docs)} DOCUMENTS SUCCESSFULLY LOADED FROM DATA  ==========")

    # === Semantic Chunking Process ===
    print("==========   SEMANTIC CHUNKING WORKING  ==========")
    
    # Initialize the semantic chunker with the provided embedding and max chunk number.
    text_splitter = SemanticChunker(embedding, number_of_chunks=MAX_CHUNK_NUMBER)
    
    # === Chunk Creation ===
    chunks = []  # Initialize an empty list to store chunks.
    for doc in docs:
        # Split the document content into smaller chunks.
        split = text_splitter.create_documents([doc.page_content])
        
        # Preserve metadata for each chunk.
        for chunk in split:
            chunk.metadata = doc.metadata  # Transfer the original metadata to the chunk.
        chunks.extend(split)  # Add the processed chunks to the list.

    print(f"==========   CHUNKS CREATED: {len(chunks)}  ==========")

    # === Create a Vector Store from Chunks ===
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embedding)  # Initialize vector store with document chunks.
    print("==========   VECTORSTORE CREATED  ==========")

    # === Return the Vector Store and Category ===
    return vectorstore, category