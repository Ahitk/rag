


## NORMAL NAIVE RETREVER EKLEMEM GEREKIYOR MU BURAYA?


import gc
import glob
import os
from langchain.docstore.document import Document
from langchain.vectorstores import Chroma
from routing import get_specific_directory

# DenseX
# Orjnal DenseX, specific_directory'den cekio
# Yedek 2 li retriever drive'da

# Sabitler
TOP_N = 20 # Ilgili specified directory'den kac tane en yakin dosyayi getirmek istedigim.
SUMMARY_FILE_PATTERN = '**/_summary.txt'

vectorstore = None
retriever = None

# Çöp toplama işlemi
gc.collect()

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

    return summaries


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
    print(f"Total embeddings calculated: {len(summary_embeddings)}")

    # Create Document objects
    for i, summary in enumerate(summaries_text):
        doc = Document(page_content=summary, metadata={'source': file_paths[i]})
        documents.append(doc)
    
    # Debug: Print size of documents list
    print(f"Total documents created: {len(documents)}")
    
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
        results = summary_retriever.get_relevant_documents(question)
        
        # Debug: Print how many results were found in this iteration
        print(f"Iteration {retries + 1}, results found: {len(results)}")

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
    print(f"Number of unique results retrieved: {len(unique_paths)}")

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
            print(f"Successfully loaded document from: {summary_path}")  # Debug: Log successful load
        except FileNotFoundError:
            print(f"Original document not found for summary: {summary_path}")  # Debug: Log missing file
        except Exception as e:
            print(f"Error loading document from {summary_path}: {e}")  # Debug: Log any other error
    
    return docs


def get_vectorstore(question, model, data_directory, embedding):
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
    print("Summary vectorstore has been cleared.")
    # En yakın özetlerin işaret ettiği orijinal dosyaları yükleyin
    docs = load_original_documents_from_summary_paths(closest_summary_files)
    # Orijinal belgelerden bir vektör mağazası ve retriever oluşturun
    vectorstore = Chroma.from_documents(documents=docs, embedding=embedding)
    return vectorstore
    #retriever = vectorstore.as_retriever()
    #return retriever














