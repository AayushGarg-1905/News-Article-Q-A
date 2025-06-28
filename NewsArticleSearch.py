import streamlit as st
from dotenv import load_dotenv
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
import pickle
import os
from sentence_transformers import SentenceTransformer
load_dotenv()  # loads GROQ_API_KEY from .env

st.title("News Article Q&A with Groq")
st.sidebar.title("Enter News URLs")


urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    if url.strip():
        urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
body_placeholder = st.empty()


if process_url_clicked:
    if not urls:
        st.sidebar.warning("Please enter at least one valid URL.")
    else:
        try:
            body_placeholder.text("Loading data from URLs...")
            loader = UnstructuredURLLoader(urls=urls)
            data = loader.load()

            body_placeholder.text("Splitting content into chunks...")
            text_splitter = RecursiveCharacterTextSplitter(
                separators=["\n\n", "\n", ".", ","],
                chunk_size=800,
                chunk_overlap=200
            )
            docs = text_splitter.split_documents(data)
            # print(docs)
            body_placeholder.text("Generating embeddings...")
            embedding_model = HuggingFaceEmbeddings(
                model_name="intfloat/e5-small-v2"
            )
            vectorstore = FAISS.from_documents(docs, embedding_model)

            body_placeholder.text("Saving vectorstore locally...")
            with open("vectorstore.pkl", "wb") as f:
                pickle.dump(vectorstore, f)

            body_placeholder.success("Processing complete! Ask your question below.")

        except Exception as e:
            st.error(f"Error processing URLs: {e}")


query = st.text_input("Ask a question based on the articles:")

if query:
    try:
        body_placeholder.text("Thinking...")

        embedding_model = HuggingFaceEmbeddings(
            model_name="intfloat/e5-small-v2"
        )
        with open("vectorstore.pkl", "rb") as f:
            vectorstore = pickle.load(f)

        llm = ChatGroq(
            temperature=0,
            model_name="llama3-8b-8192",
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            return_source_documents=True
        )

        result = qa_chain({"query": query})
        body_placeholder.empty()

        st.subheader("Answer:")
        st.write(result["result"])

        with st.expander("See source chunks"):
            for i, doc in enumerate(result["source_documents"]):
                st.markdown(f"**Chunk {i+1}:**")
                st.markdown(doc.page_content)
                st.markdown("---")

    except Exception as e:
        st.error(f"Failed to generate answer: {e}")
