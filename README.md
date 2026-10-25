# Lecture Notes & Course Material Q&A RAG Assistant

Single-source RAG application built for Phase 1 of the AI Engineering Masters Program.

## Tech Stack & Architecture
- **Frontend:** Streamlit (`app.py`)
- **Orchestration:** LangChain LCEL
- **Embeddings & Vector Store:** Hugging Face `sentence-transformers/all-MiniLM-L6-v2` with FAISS (`faiss-cpu`)
- **Ingestion:** PyPDF loader with page-level structural preservation

## Setup Instructions
1. Clone repo: `git clone https://github.com/Vaibhav8881/Lecture-Notes-RAG-Assistant.git`
2. Install dependencies: `pip install -r requirement.txt`
3. Run app: `streamlit run app.py`
4. 
