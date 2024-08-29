import os
import re
import unicodedata
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from chromadb import Client
from chromadb.config import Settings
import streamlit as st

# .env dosyasını yükle
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Veri Yükleme ve Ayıklama

data_directory = "rag_data/website/organized_data"

def extract_qa_pairs(text):
    """
    Bir txt dosyasındaki tüm soru-cevap çiftlerini çıkartır.
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
    Klasördeki tüm txt dosyalarını yükler ve bunları kategorilere göre sınıflandırır.
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
print(f"Yüklendi {sum(len(v) for v in categorized_data.values())} adet soru-cevap çifti.")

# Embeddings ve Veritabanına Yükleme

# OpenAI Embeddings başlat
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# ChromaDB Client oluştur
chroma_client = Client(Settings())

def normalize_text(text):
    """
    Özel karakterleri kaldırarak metni normalleştirir.
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
    Bir metni vektöre dönüştürür.
    """
    return embeddings.embed_query(text)

def valid_collection_name(name):
    """
    Koleksiyon ismini geçerli bir formata dönüştürür.
    """
    name = normalize_text(name.lower())
    name = name.replace(" ", "_").replace("&", "and").replace(",", "")
    if len(name) < 3:
        name = name + "_collection"
    return name[:63]

def create_and_populate_database(categorized_data):
    """
    Kategorilere göre veritabanına vektörleri ekler.
    """
    existing_collections = set(chroma_client.list_collections())
    
    for category, qa_pairs in categorized_data.items():
        valid_name = valid_collection_name(category)

        # Eğer koleksiyon mevcut değilse oluştur
        if valid_name not in existing_collections:
            print(f"Koleksiyon '{valid_name}' bulunamadı, oluşturulacak.")
            collection = chroma_client.create_collection(name=valid_name)
        else:
            print(f"Mevcut koleksiyon '{valid_name}' kullanılıyor.")
            collection = chroma_client.get_collection(name=valid_name)
        
        for idx, qa_pair in enumerate(qa_pairs):
            question = qa_pair["question"]
            answer = qa_pair["answer"]
            vector = create_embeddings(question)
            metadata = {"question": question, "answer": answer}
            unique_id = f"{valid_name}_{idx}"

            # Koleksiyona veriyi ekle
            collection.add(ids=[unique_id], documents=[question], embeddings=[vector], metadatas=[metadata])

        print(f"{len(qa_pairs)} adet soru-cevap çifti '{valid_name}' kategorisine eklendi.")

create_and_populate_database(categorized_data)

# Streamlit Uygulaması

def initialize_qa_chain(category):
    """
    Belirli bir kategori için RAG Retrieval QA zincirini başlatır.
    """
    collection_name = valid_collection_name(category)
    vectorstore = Chroma(collection_name, embedding_function=embeddings.embed_query)

    llm = OpenAI(model="gpt-4", openai_api_key=OPENAI_API_KEY)

    prompt_template = PromptTemplate(
        template="Soru: {question}\n\nCevap:",
        input_variables=["question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        prompt=prompt_template
    )
    
    return qa_chain

def main():
    st.title("RAG QA Uygulaması")
    
    # Kategori seçim listesi
    categories = ["magentaeins", "hilfe_bei_storungen", "gerate_and_zubehor", "others", 
                  "tv_collection", "vertrag_and_rechnung", "internet_and_telefonie", 
                  "apps_and_dienste", "mobilfunk"]
    
    st.sidebar.header("Ayarlar")
    selected_category = st.sidebar.selectbox("Kategori Seçin", categories)
    
    st.sidebar.write(f"Seçilen Kategori: {selected_category}")
    
    question = st.text_input("Sorunuzu buraya yazın:")
    
    if st.button("Soru Sor"):
        if question:
            qa_chain = initialize_qa_chain(selected_category)
            response = qa_chain({"question": question})
            st.write("Cevap:", response)
        else:
            st.write("Lütfen bir soru girin.")

if __name__ == "__main__":
    main()