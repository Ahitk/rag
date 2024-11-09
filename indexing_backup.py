import gc
import glob
import os
import random
from langchain.docstore.document import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import CharacterTextSplitter
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_chroma import Chroma
from routing import get_specific_directory


# Constants
TOP_N = 30 # Ilgili specified directory'den kac tane en yakin dosyayi getirmek istedigim.
HYBRID_VECTORSTORE_WEIGHT = 0.5
MAX_CHUNK_NUMBER = 5 
MAX_DOCUMENT_NUMBER_K = 10
SUMMARY_FILE_PATTERN = '**/_summary.txt'

vectorstore = None
retriever = None

# Çöp toplama işlemi
gc.collect()

# DENSE X 

# ================================ USE FILE NAME ==================================

# Constants
SUMMARY_FILENAME_PATTERN = '**/_summary.txt'
RETRIEVAL_TOP_N = 30  # Number of closest documents to retrieve

def extract_summaries_from_files(summary_dir):
    """
    Extracts summaries from _summary.txt files in the specified directory, using only file names.
    Returns a dictionary where keys are file names and values are summaries.
    """
    summary_file_paths = glob.glob(os.path.join(summary_dir, SUMMARY_FILENAME_PATTERN), recursive=True)
    file_summary_map = {}

    for summary_file in summary_file_paths:
        with open(summary_file, 'r') as f:
            content = f.read()

        # Split content into chunks using "=== Chunk ===" delimiter
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


def initialize_summary_vectorstore(file_summaries, embedding_model):
    """
    Initializes a Chroma vector store from provided summaries using the embedding model.
    Each Document object in the vector store includes only the summary text.
    """
    summary_documents = [Document(page_content=summary) for summary in file_summaries.values()]

    # Create a Chroma vector store from the list of summary documents
    summary_vector_store = Chroma.from_documents(documents=summary_documents, embedding=embedding_model)
    return summary_vector_store


def retrieve_closest_filenames(user_question, summary_retriever, summary_map, top_n=RETRIEVAL_TOP_N):
    """
    Finds the closest summaries based on the user's question using the vector retriever.
    Ensures that only unique file names are returned.
    """
    retrieved_filenames = []
    seen_files = set()
    retries = 0

    while len(retrieved_filenames) < top_n and retries < 5:
        results = summary_retriever.invoke(user_question)
        
        if results is None:
            print(f"No results retrieved for question: {user_question}")
            break

        for result in results:
            summary_content = result.page_content
            file_name = next((name for name, summary in summary_map.items() if summary == summary_content), None)
            
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


def load_original_docs_by_filenames(file_names, doc_directory):
    """
    Loads original .txt documents based on a list of file names within a specified directory.
    Returns a list of Document objects with content.
    """
    loaded_documents = []
    for file_name in file_names:
        # Construct the file path based on the file name and doc directory
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


def generate_final_vectorstore_with_chunks(user_question, doc_directory, embedding_model):
    """
    Generates a Chroma vector store by loading summaries, retrieving relevant summaries,
    loading original documents, and applying semantic chunking for enhanced retrieval.
    """
    # Load summaries from _summary.txt files
    file_summaries = extract_summaries_from_files(doc_directory)
    
    # Create a Chroma vector store from the summaries
    summary_vectorstore = initialize_summary_vectorstore(file_summaries, embedding_model)
    
    # Create a retriever from the summary vector store
    summary_retriever = summary_vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_TOP_N})
    
    # Retrieve the closest summaries and get corresponding file names
    closest_file_names = retrieve_closest_filenames(user_question, summary_retriever, file_summaries, top_n=RETRIEVAL_TOP_N)
    
    # Clear the summary vector store after retrieval
    summary_vectorstore.delete_collection()
    
    # Load the original documents using the retrieved file names
    original_documents = load_original_docs_by_filenames(closest_file_names, doc_directory)
    print(f"========== {len(original_documents)} DOCUMENTS SUCCESSFULLY LOADED ==========")

    print("========== SEMANTIC CHUNKING IN PROGRESS ==========")
    
    # Apply semantic chunking to split the documents for better retrieval granularity
    semantic_splitter = SemanticChunker(embedding_model)
    document_chunks = []
    for doc in original_documents:
        split_chunks = semantic_splitter.create_documents([doc.page_content])
        document_chunks.extend(split_chunks)
    print(f"========== TOTAL CHUNKS CREATED: {len(document_chunks)} ==========")

    # Create a final vector store from the document chunks
    final_chunked_vectorstore = Chroma.from_documents(documents=document_chunks, embedding=embedding_model)
    print("========== FINAL VECTOR STORE WITH CHUNKS CREATED ==========")
    
    return final_chunked_vectorstore

def generate_final_vectorstore_with_character_splitter(user_question, doc_directory, embedding_model):
    """
    Generates a Chroma vector store by loading summaries, retrieving relevant summaries,
    loading original documents, and applying semantic chunking for enhanced retrieval.
    """
    # Load summaries from _summary.txt files
    file_summaries = extract_summaries_from_files(doc_directory)
    
    # Create a Chroma vector store from the summaries
    summary_vectorstore = initialize_summary_vectorstore(file_summaries, embedding_model)
    
    # Create a retriever from the summary vector store
    summary_retriever = summary_vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_TOP_N})
    
    # Retrieve the closest summaries and get corresponding file names
    closest_file_names = retrieve_closest_filenames(user_question, summary_retriever, file_summaries, top_n=RETRIEVAL_TOP_N)
    
    # Clear the summary vector store after retrieval
    summary_vectorstore.delete_collection()
    
    # Load the original documents using the retrieved file names
    original_documents = load_original_docs_by_filenames(closest_file_names, doc_directory)
    print(f"========== {len(original_documents)} DOCUMENTS SUCCESSFULLY LOADED ==========")

    print("========== CHARACTER SPLITTING IN PROGRESS ==========")
    
    # Character splitting without separator
    text_splitter = CharacterTextSplitter(
        separator='',
        chunk_size=500,
        chunk_overlap=50,
    )
    
    document_chunks = []
    for doc in original_documents:
        split_chunks = text_splitter.create_documents([doc.page_content])
        document_chunks.extend(split_chunks)
    print(f"========== TOTAL CHUNKS CREATED: {len(document_chunks)} ==========")

    # Create a final vector store from the document chunks
    final_chunked_vectorstore = Chroma.from_documents(documents=document_chunks, embedding=embedding_model)
    print("========== FINAL VECTOR STORE WITH CHUNKS CREATED ==========")
    
    return final_chunked_vectorstore

#================================== USE FILE PATH =====================================

def load_summaries(data_directory):
    """
    Summarize the content of _summary.txt files in the given directory.
    """
    summary_files = glob.glob(os.path.join(data_directory, SUMMARY_FILE_PATTERN), recursive=True)
    summaries = {}
    
    for file in summary_files:
        with open(file, 'r') as f:
            content = f.read()
        
        chunks = content.split("=== Chunk ===")
        for chunk in chunks:
            if "File path:" in chunk and "File summary:" in chunk:
                try:
                    lines = chunk.split('\n')
                    file_path_line = [line for line in lines if "File path:" in line]
                    summary_line = [line for line in lines if "File summary:" in line]

                    if file_path_line and summary_line:
                        file_path = file_path_line[0].split("File path:")[1].strip()
                        summary_text = summary_line[0].split("File summary:")[1].strip()
                        summaries[file_path] = summary_text
                except IndexError:
                    print(f"Warning: Skipping chunk due to formatting issues in file: {file}")

    return summaries, data_directory


def create_chroma_vectorstore(summaries, embedding):
    """
    Create a Chroma vectorstore from the provided summaries.
    """
    documents = []
    summaries_text = list(summaries.values())
    file_paths = list(summaries.keys())
    
    # Embed summaries in batch
    summary_embeddings = embedding.embed_documents(summaries_text)
    
    # Debug: Print size of embeddings and a sample embedding
    #print(f"Total embeddings calculated: {len(summary_embeddings)}")

    # Create Document objects
    for i, summary in enumerate(summaries_text):
        doc = Document(page_content=summary, metadata={'source': file_paths[i]})
        documents.append(doc)
    
    # Debug: Print size of documents list
    #print(f"Total documents created: {len(documents)}")
    
    # Create Chroma vectorstore from documents
    summary_vectorstore = Chroma.from_documents(documents=documents, embedding=embedding)
    return summary_vectorstore


def find_closest_summaries_with_chroma(question, summary_retriever, top_n=TOP_N):
    """
    Finds the closest summary files based on the user's question using the Chroma retriever.
    Ensures that only unique file paths are returned, with no duplicates. If there aren't
    enough unique results, it keeps searching until it finds `top_n` unique results.
    """
    unique_paths = []
    seen_files = set()
    retries = 0  # To prevent infinite loops in case something goes wrong

    while len(unique_paths) < top_n and retries < 5:  # Limit retries to 5 to avoid infinite loops
        # Get results from the retriever
        #results = summary_retriever.get_relevant_documents(question)
        results = summary_retriever.invoke(question)
        # Debug: Print how many results were found in this iteration
        #print(f"Iteration {retries + 1}, results found: {len(results)}")

        for result in results:
            file_path = result.metadata['source']
            
            # Check if the file path has already been added
            if file_path not in seen_files:
                unique_paths.append(file_path)
                seen_files.add(file_path)
            
            # Stop once we have the desired number of unique paths
            if len(unique_paths) >= top_n:
                break
        
        retries += 1  # Increment retry counter in case we need to search again

    # Debug: Print how many unique results were retrieved in total
    print(f"==========   NUMBER OF DOCUMENTS RETRIEVED: {len(unique_paths)}   ==========")

    # If after retries we still don't have enough results, warn the user
    if len(unique_paths) < top_n:
        print(f"Warning: Only {len(unique_paths)} unique results were found after {retries} iterations.")

    return unique_paths


def load_original_documents_from_summary_paths(summary_paths):
    """
    Load the original documents based on the summary file paths.
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
            #print(f"Successfully loaded document from: {summary_path}")  # Debug: Log successful load
        except FileNotFoundError:
            print(f"Original document not found for summary: {summary_path}")  # Debug: Log missing file
        except Exception as e:
            print(f"Error loading document from {summary_path}: {e}")  # Debug: Log any other error
    
    return docs
 
# DIKKAT! BURADA SEMANTIC CHUNKIG KULLANILIYOR.
def get_vectorstore(question, model, data_directory, embedding):
    # Özetleri yükleyin
    summaries, category = load_summaries(get_specific_directory(question, model, data_directory))
    # Chroma vektör mağazasını oluşturun
    summary_vectorstore = create_chroma_vectorstore(summaries, embedding)
    # Chroma'dan bir retriever oluşturun
    summary_retriever = summary_vectorstore.as_retriever(search_kwargs={"k": TOP_N})    
    # En yakın özetleri bulun
    closest_summary_files = find_closest_summaries_with_chroma(question, summary_retriever, top_n=TOP_N)
    # Chroma vectorstore'u temizleyin
    summary_vectorstore.delete_collection()  # Bu tüm vektörleri silecek
    # En yakın özetlerin işaret ettiği orijinal dosyaları yükleyin
    docs = load_original_documents_from_summary_paths(closest_summary_files)
    print(f"==========   {len(docs)} DOCUMENTS SUCCESSFULLY LOADED FROM DATA  ==========")

    print("==========   SEMANTIC CHUNKING WORKING  ==========")
    
    # SEMANTIC CHUNKING
    text_splitter = SemanticChunker(embedding)
    
    # Her bir orijinal belgeyi daha küçük parçalara bölün ve hepsini bir listeye ekleyin
    chunks = []
    for doc in docs:
        split = text_splitter.create_documents([doc.page_content])  # İçeriği küçük parçalara ayır
        # Orijinal metadata'yı her bir parça için koru
        for chunk in split:
            chunk.metadata = doc.metadata  # Metadata’yı koruyarak ekle
        chunks.extend(split)
    print(f"==========   CHUNKS CREATED: {len(chunks)}  ==========")

    # Chunk'lerden bir vektör mağazası oluşturun
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embedding)
    print("==========   VECTORSTORE CREATED  ==========")
    return vectorstore #,category


#============================== FOR TEST PURPOSES ONLY ================================

def get_vectorstore_semantic_chunking_no_summary(test_directory, embedding):
    
    all_txt_files = glob.glob(os.path.join(test_directory, "*.txt"))

    # Seçilen dosyaların içeriklerini oku ve birleştir
    all_texts = []
    for file_path in all_txt_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            all_texts.append(f.read())

    print("==========   SEMANTIC CHUNKING WORKING  ==========")
    
    # SEMANTIC CHUNKING
    text_splitter = SemanticChunker(embedding)
    chunks = text_splitter.create_documents(all_texts)

    print(f"==========   CHUNKS CREATED: {len(chunks)}  ==========")

    # Chunk'lerden bir vektör mağazası oluşturun
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embedding)
    print("==========   VECTORSTORE CREATED  ==========")
    return vectorstore

#============================== END =====================================================


###### ORIGINAL OLD WITHOUT SEMANTIC CHUNKING
def get_vectorstore_without_chunking(question, model, data_directory, embedding):
    # Özetleri yükleyin
    summaries = load_summaries(get_specific_directory(question, model, data_directory))
    # Chroma vektör mağazasını oluşturun
    summary_vectorstore = create_chroma_vectorstore(summaries, embedding)
    # Chroma'dan bir retriever oluşturun
    summary_retriever = summary_vectorstore.as_retriever(search_kwargs={"k": TOP_N})    
    # En yakın özetleri bulun
    closest_summary_files = find_closest_summaries_with_chroma(question, summary_retriever, top_n=TOP_N)
    # Clear Chroma vectorstore after use
    summary_vectorstore.delete_collection()  # This will delete all vectors in the collection
    #print("Summary vectorstore has been cleared.")
    # En yakın özetlerin işaret ettiği orijinal dosyaları yükleyin
    docs = load_original_documents_from_summary_paths(closest_summary_files)
    print(f"==========   {len(docs)} DOCUMENTS SUCCESSFULLY LOADED FROM DATA  ==========")
    # Orijinal belgelerden bir vektör mağazası ve retriever oluşturun
    vectorstore = Chroma.from_documents(documents=docs, embedding=embedding)
    print("==========   VECTORSTORE CREATED  ==========")

    return vectorstore

def get_hybrid_semantic_retriever(question, model, data_directory, embedding):
    # Özetleri yükleyin
    summaries = load_summaries(get_specific_directory(question, model, data_directory))
    # Chroma vektör mağazasını oluşturun
    summary_vectorstore = create_chroma_vectorstore(summaries, embedding)
    # Chroma'dan bir retriever oluşturun
    summary_retriever = summary_vectorstore.as_retriever(search_kwargs={"k": TOP_N})    
    # En yakın özetleri bulun
    closest_summary_files = find_closest_summaries_with_chroma(question, summary_retriever, top_n=TOP_N)
    # Chroma vectorstore'u temizleyin
    summary_vectorstore.delete_collection()  # Bu tüm vektörleri silecek
    # En yakın özetlerin işaret ettiği orijinal dosyaları yükleyin
    docs = load_original_documents_from_summary_paths(closest_summary_files)
    print(f"==========   DOCUMENTS SUCCESSFULLY LOADED FROM DATA  ==========")

    print("==========   SEMANTIC CHUNKING WORKING  ==========")
    # SEMANTIC CHUNKING
    text_splitter = SemanticChunker(embedding, number_of_chunks=MAX_CHUNK_NUMBER)
    
    # Her bir orijinal belgeyi daha küçük parçalara bölün ve hepsini bir listeye ekleyin
    chunks = []
    for doc in docs:
        split = text_splitter.create_documents([doc.page_content])  # İçeriği küçük parçalara ayır
        # Orijinal metadata'yı her bir parça için koru
        for chunk in split:
            chunk.metadata = doc.metadata  # Metadata’yı koruyarak ekle
        chunks.extend(split)
    print(f"==========   CHUNKS CREATED: {len(chunks)}  ==========")

    # Chunk'lerden bir vektör mağazası oluşturun
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embedding)
    print("==========   VECTORSTORE CREATED  ==========")
    semantic_retriever = vectorstore.as_retriever()
    keyword_retriever = BM25Retriever.from_documents(chunks)
    hybrid_retriever = EnsembleRetriever(retrievers=[keyword_retriever, semantic_retriever], weights=[1-HYBRID_VECTORSTORE_WEIGHT, HYBRID_VECTORSTORE_WEIGHT])
    print("==========   HYBRID SEARCH FINISHED  ==========")
    return hybrid_retriever




# Naive RAG - Semantic Search - Character Splitting
def get_vectorstore_naive_semantic(test_directory, embedding):
    # Klasörden tüm txt dosyalarını topla

    all_txt_files = glob(os.path.join(test_directory, "*.txt"))
    
    # Eğer 50'den fazla dosya varsa rastgele 50 tanesini seç
    selected_files = random.sample(all_txt_files, min(50, len(all_txt_files)))
    
    # Seçilen dosyaların içeriklerini oku ve birleştir
    all_texts = []
    for file_path in selected_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            all_texts.append(f.read())
    
    # Text splitter ile içerikleri chunk'lara ayır
    text_splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=200,
        chunk_overlap=30,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.create_documents(all_texts)
    
    # Chunk'lerden bir vektör mağazası oluştur
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embedding)
    print("==========   VECTORSTORE CREATED  ==========")
    
    # Kategori ismini belirle (directory'nin son kısmı)
    category = os.path.basename(test_directory.rstrip('/'))
    
    return vectorstore, category



# Advanced RAG	- Semantic Search -	Dense X	- Semantic Splitting - Fusion -	Logical Routing
# DIKKAT! BURADA DIREKT ILGILI DIRECTORY'YI ALIYOR, DIRECTORY ROUTING YOK.
def get_vectorstore_semantic(question, test_directory, embedding):
    # Özetleri yükleyin
    summaries, category = load_summaries(test_directory)
    # Chroma vektör mağazasını oluşturun
    summary_vectorstore = create_chroma_vectorstore(summaries, embedding)
    # Chroma'dan bir retriever oluşturun
    summary_retriever = summary_vectorstore.as_retriever(search_kwargs={"k": TOP_N})    
    # En yakın özetleri bulun
    closest_summary_files = find_closest_summaries_with_chroma(question, summary_retriever, top_n=TOP_N)
    # Chroma vectorstore'u temizleyin
    summary_vectorstore.delete_collection()  # Bu tüm vektörleri silecek
    # En yakın özetlerin işaret ettiği orijinal dosyaları yükleyin
    docs = load_original_documents_from_summary_paths(closest_summary_files)
    print(f"==========   {len(docs)} DOCUMENTS SUCCESSFULLY LOADED FROM DATA  ==========")

    print("==========   SEMANTIC CHUNKING WORKING  ==========")
    
    # SEMANTIC CHUNKING
    text_splitter = SemanticChunker(embedding, number_of_chunks=MAX_CHUNK_NUMBER)
    
    # Her bir orijinal belgeyi daha küçük parçalara bölün ve hepsini bir listeye ekleyin
    chunks = []
    for doc in docs:
        split = text_splitter.create_documents([doc.page_content])  # İçeriği küçük parçalara ayır
        # Orijinal metadata'yı her bir parça için koru
        for chunk in split:
            chunk.metadata = doc.metadata  # Metadata’yı koruyarak ekle
        chunks.extend(split)
    print(f"==========   CHUNKS CREATED: {len(chunks)}  ==========")

    # Chunk'lerden bir vektör mağazası oluşturun
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embedding)
    print("==========   VECTORSTORE CREATED  ==========")
    return vectorstore, category