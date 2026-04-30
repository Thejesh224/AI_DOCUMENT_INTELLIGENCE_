# 📄 AI Document Intelligence System

## 🔍 Overview
This is an AI-powered web application that allows users to upload PDF documents and ask questions. The system uses Retrieval-Augmented Generation (RAG) to provide accurate answers.

---

## 🚀 Features
- 🔐 Login & Create Account
- 👥 User count tracking
- 📄 Upload PDF documents
- 💬 Chat-based Q&A system
- 🤖 AI-powered responses using LLM
- ⚡ Fast retrieval using FAISS

---

## 🧠 Tech Stack
- Python
- Streamlit
- LangChain
- FAISS
- HuggingFace Embeddings
- OpenAI API

---

## ⚙️ How It Works
1. Upload a PDF
2. Text is extracted and split into chunks
3. Chunks are converted into embeddings
4. Stored in FAISS vector database
5. User asks a question
6. Relevant chunks are retrieved
7. LLM generates the final answer

---

## ▶️ Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
