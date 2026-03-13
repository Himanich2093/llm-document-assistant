import streamlit as st
from dotenv import load_dotenv
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_community.embeddings import HuggingFaceEmbeddings
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

@st.cache_resource
def process_pdf(uploaded_file):

    uploaded_file.seek(0)

    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    loader = PyPDFLoader("temp.pdf")
    pages = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    docs = text_splitter.split_documents(pages)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.from_documents(docs, embeddings)

    return vector_store

st.set_page_config(page_title="LLM Document Assistant")
st.title("📚 Chat With Your Documents")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:

    uploaded_file.seek(0)

    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    st.success("PDF uploaded successfully!")

    # Load PDF
    loader = PyPDFLoader("temp.pdf")
    pages = loader.load()

    # Split text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    docs = text_splitter.split_documents(pages)

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    # Create vector DB
    try:
        vector_store = process_pdf(uploaded_file)
        retriever = vector_store.as_retriever()

    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        st.stop()

    # Load LLM
    model = genai.GenerativeModel("gemini-flash-latest")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask a question about the document")

    if prompt:

        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })

        with st.chat_message("user"):
            st.markdown(prompt)

        # Retrieve relevant chunks
        relevant_docs = retriever.invoke(prompt)

        context = "\n\n".join([doc.page_content for doc in relevant_docs])

        # Ask LLM
        response = model.generate_content(
            f"""
            Use the context below to answer the question.

            Context:
            {context}

            Question:
            {prompt}
            """
        )

        answer = response.text

        with st.chat_message("assistant"):
            st.markdown(answer)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })