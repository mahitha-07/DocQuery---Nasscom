import os
from dotenv import load_dotenv
from typing import List
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")

def get_pdf_text(pdf_files):
    full_text = ""
    for pdf in pdf_files:
        pdf_reader = PdfReader(pdf)
        for page_num, page in enumerate(pdf_reader.pages, start=1):
            page_text = page.extract_text()
            if page_text:
                full_text += f"--- Page {page_num} ---\n" + page_text + "\n"
    return full_text

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    return splitter.split_text(text)

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=google_api_key
    )
    vector_store = FAISS.from_texts(text_chunks, embeddings)
    return vector_store

def get_conversational_chain(vector_store):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-001",
        google_api_key=google_api_key
    )
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm, retriever=vector_store.as_retriever(), memory=memory
    )
    return conversation_chain

def user_input(question, conversation_chain):
    result = conversation_chain({"question": question})
    chat_history = result.get("chat_history", [])
    for i, message in enumerate(chat_history):
        if i % 2 == 0:
            st.write("Human: " + message.content)
        else:
            st.write("Bot: " + message.content)

def main():
    st.set_page_config(page_title="DocuQuery: AI-Powered PDF Knowledge Assistant")
    st.header("DocuQuery: AI-Powered PDF Knowledge Assistant")

    st.sidebar.title("Settings")
    st.sidebar.subheader("Upload your Documents")
    pdf_files = st.sidebar.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
    process_button = st.sidebar.button("Process")

    if process_button:
        if pdf_files:
            with st.spinner("Processing documents..."):
                text = get_pdf_text(pdf_files)
                text_chunks = get_text_chunks(text)
                vector_store = get_vector_store(text_chunks)
                st.session_state["conversation_chain"] = get_conversational_chain(vector_store)
            st.success("Documents processed successfully!")
        else:
            st.sidebar.error("Please upload at least one PDF file.")

    question = st.text_input("Enter your question:")
    if question:
        if "conversation_chain" in st.session_state:
            user_input(question, st.session_state["conversation_chain"])
        else:
            st.error("Please process the documents first!")

if __name__ == "__main__":
    main()
