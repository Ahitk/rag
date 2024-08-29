import streamlit as st
import os
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

# .env dosyasını yükle
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def valid_collection_name(name):
    """
    Koleksiyon ismini geçerli bir formata dönüştürür.
    """
    name = name.lower().replace(" ", "_").replace("&", "and").replace(",", "")
    if len(name) < 3:
        name += "_collection"
    return name[:63]

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
    
    # Kategori seçim listesi - Burayı mevcut koleksiyon isimleri ile doldurabilirsiniz
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