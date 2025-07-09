import streamlit as st
import ollama
import os
import tempfile
import shutil
import logging
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="RAG via Notebook", layout="wide")
st.title("📄 PDF RAG App (Notebook Version)")

uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

try:
    models = ollama.list()["models"]
    model_names = [m["name"] for m in models]
except Exception as e:
    st.error("Ollama is not running. Please start `ollama serve` in your terminal.")
    st.stop()

selected_model = st.selectbox("Select Ollama Model", model_names)

if uploaded_file:

    with tempfile.TemporaryDirectory() as tmp_dir:
        pdf_path = os.path.join(tmp_dir, uploaded_file.name)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        st.success("PDF uploaded and saved!")

        st.info("🔍 Loading and splitting PDF...")
        loader = UnstructuredPDFLoader(pdf_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
        chunks = splitter.split_documents(docs)

        st.success(f"✅ Split into {len(chunks)} chunks.")

        st.info("🔗 Generating embeddings and vector DB...")
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        vectordb = Chroma.from_documents(chunks, embeddings, collection_name="notebookRAG")

        st.success("✅ Vector DB ready!")

        llm = ChatOllama(model=selected_model, temperature=0.0)
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectordb.as_retriever(),
            return_source_documents=True
        )

        user_query = st.text_input("Ask a question about the PDF:")

        if user_query:
            with st.spinner("Generating answer..."):
                result = qa_chain(user_query)
                st.markdown(f"**Answer:** {result['result']}")
                with st.expander("🔎 Sources"):
                    for doc in result["source_documents"]:
                        st.markdown(doc.page_content[:400] + "...")


