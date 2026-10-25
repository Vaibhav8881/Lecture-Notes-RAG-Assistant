import os
import streamlit as st
from langchain_openai import ChatOpenAI
from rag_engine import process_pdf_to_vectorstore, create_rag_chain

st.set_page_config(page_title="Lecture Notes Q&A Assistant", page_icon="📚", layout="wide")
st.title("📚 Lecture Notes & Course Material Q&A Assistant")
st.markdown("Upload your lecture notes PDF and ask questions grounded directly in the text with page citations.")

with st.sidebar:
    st.header("⚙️ Configuration & Upload")
    api_key = st.text_input("Enter OpenAI API Key", type="password")
    uploaded_file = st.file_uploader("Upload Lecture PDF", type=["pdf"])
    st.markdown("---")
    st.markdown("**Phase 1: RAG Foundations**\n- Embeddings: `all-MiniLM-L6-v2`\n- Vector Store: `FAISS`\n- LCEL Chain")

if uploaded_file is not None:
    if "vector_store" not in st.session_state or st.session_state.get("current_file") != uploaded_file.name:
        with st.spinner("Processing PDF and building local FAISS index..."):
            vstore, num_pages, num_chunks = process_pdf_to_vectorstore(uploaded_file)
            st.session_state["vector_store"] = vstore
            st.session_state["current_file"] = uploaded_file.name
        st.success(f"Indexed {num_pages} pages into {num_chunks} chunks successfully!")

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about your lecture notes..."):
        if not api_key:
            st.error("Please enter your OpenAI API key in the sidebar.")
        else:
            st.session_state["messages"].append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Retrieving context and generating answer..."):
                    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2, api_key=api_key)
                    rag_chain = create_rag_chain(st.session_state["vector_store"], llm)
                    response = rag_chain.invoke({"input": prompt})
                    answer = response["answer"]
                    source_docs = response.get("context", [])

                    st.markdown(answer)
                    if source_docs:
                        with st.expander("📄 View Source Chunk Citations"):
                            for i, doc in enumerate(source_docs):
                                page_num = doc.metadata.get("page", 0) + 1
                                st.markdown(f"**Source {i+1} (Page {page_num}):** `{doc.page_content[:200]}...`")

                    st.session_state["messages"].append({"role": "assistant", "content": answer})
else:
    st.info("👈 Please enter your API key and upload a PDF document in the sidebar to begin.")
