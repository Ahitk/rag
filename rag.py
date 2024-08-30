import os
import re
import unicodedata
from dotenv import load_dotenv
from langchain import OpenAI, create_retrieval_chain
from langchain.chains import ChatPromptTemplate, create_stuff_documents_chain
from langchain_chains import create_retrieval_chain
from langchain_chroma import Chroma
from chromadb import Client
from chromadb.config import Settings

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Data Loading and Extraction

data_directory = "rag_data/website/organized_data"

def extract_qa_pairs(text):
    """
    Extracts all question-answer pairs from a txt file.
    """
    qa_pairs = []
    pattern = r"(\d+)\.\s*Question:(.*?)\n\s*Answer:(.*?)(?=\n\d+\.|\Z)"
    matches = re.findall(pattern, text, re.DOTALL)
    
    for match in matches:
        question = match[1].strip()
        answer = match[2].strip()
        qa_pairs.append({"question": question, "answer": answer})
    
    return qa_pairs

def load_data(directory):
    """
    Loads all txt files in the folder and classifies them by category.
    """
    categorized_data = {}
    
    for category in os.listdir(directory):
        category_path = os.path.join(directory, category)
        if os.path.isdir(category_path):
            categorized_data[category] = []
            
            for file_name in os.listdir(category_path):
                if file_name.endswith(".txt"):
                    file_path = os.path.join(category_path, file_name)
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        qa_pairs = extract_qa_pairs(content)
                        categorized_data[category].extend(qa_pairs)
    
    return categorized_data

categorized_data = load_data(data_directory)
print(f"Loaded {sum(len(v) for v in categorized_data.values())} question-answer pairs.")

# Embeddings and Database Loading

# Initialize OpenAI Embeddings
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Initialize ChromaDB Client
settings = Settings()
chroma_client = Client(settings=settings)

def normalize_text(text):
    """
    Normalizes text by removing special characters.
    """
    replacements = {
        'ö': 'o', 'ü': 'u', 'ä': 'a', 'ß': 'ss', 
        'ğ': 'g', 'ç': 'c', 'ş': 's', 'ı': 'i'
    }
    for search, replace in replacements.items():
        text = text.replace(search, replace)
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')

def create_embeddings(text):
    """
    Converts text into a vector.
    """
    return embeddings.embed_query(text)

def valid_collection_name(name):
    """
    Converts the collection name into a valid format.
    """
    name = normalize_text(name.lower())
    name = name.replace(" ", "_").replace("&", "and").replace(",", "")
    if len(name) < 3:
        name = name + "_collection"
    return name[:63]

def create_and_populate_database(categorized_data):
    """
    Adds vectors to the database by categories.
    """
    # List existing collections
    existing_collections = {c.name for c in chroma_client.list_collections()}
    print(f"Existing Collections: {existing_collections}")
    
    for category, qa_pairs in categorized_data.items():
        valid_name = valid_collection_name(category)

        if valid_name not in existing_collections:
            print(f"Collection '{valid_name}' not found, creating.")
            collection = chroma_client.create_collection(name=valid_name)
        else:
            print(f"Using existing collection '{valid_name}'.")
            collection = chroma_client.get_collection(name=valid_name)

        # Correct way to handle existing documents: fetch IDs in a different way
        existing_docs = collection.get()
        existing_ids = {doc["id"] for doc in existing_docs}

        new_ids = set()
        for idx, qa_pair in enumerate(qa_pairs):
            question = qa_pair["question"]
            answer = qa_pair["answer"]
            vector = create_embeddings(question)
            metadata = {"question": question, "answer": answer}
            unique_id = f"{valid_name}_{idx}"

            # Skip if the ID already exists
            if unique_id in existing_ids:
                continue

            # Add data to the collection
            collection.add(ids=[unique_id], documents=[question], embeddings=[vector], metadatas=[metadata])
            new_ids.add(unique_id)

        print(f"Added {len(new_ids)} new question-answer pairs to the '{valid_name}' category.")

create_and_populate_database(categorized_data)

def initialize_qa_chain():
    """
    Initializes the RAG Retrieval QA chain for all categories.
    """
    retrievers = []
    for category in categorized_data.keys():
        collection_name = valid_collection_name(category)
        vectorstore = Chroma(collection_name, embedding_function=embeddings.embed_query)
        retrievers.append(vectorstore.as_retriever())

    # Combine all retrievers into a single retriever
    combined_retriever = Chroma.combine_retrievers(retrievers)

    llm = OpenAI(model="gpt-4", openai_api_key=OPENAI_API_KEY)

    system_prompt = (
        "Use the given context to answer the question. "
        "If you don't know the answer, say you don't know. "
        "Use three sentence maximum and keep the answer concise. "
        "Context: {context}"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    chain = create_retrieval_chain(retriever=combined_retriever, combine_docs_chain=question_answer_chain)
    
    return chain

def ask_question(chain):
    """
    Asks a question via the console and retrieves an answer.
    """
    while True:
        question = input("Enter your question (or type 'exit' to quit): ")
        if question.lower() == 'exit':
            break
        
        response = chain.invoke({"input": question})
        print("Answer:", response.get("result", "No answer found"))

# Main Execution
if __name__ == "__main__":
    print("Initializing QA Chain...")
    qa_chain = initialize_qa_chain()
    print("QA Chain is ready. You can now ask questions.")
    ask_question(qa_chain)