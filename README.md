# 💼 Role-Based RAG Assistant

This is a **Role-Based Retrieval-Augmented Generation (RAG)** system built with **LangChain**, **FastAPI**, **Streamlit**, and **Groq's LLMs**. It classifies user queries, retrieves relevant document chunks based on user roles, and uses an LLM to synthesize answers strictly from those documents.

---

## 📌 Features

- 🔐 **Role-Based Access**: Ensures users can only view content relevant to their role (e.g., HR, Engineering, Finance).
- 🔍 **Hybrid Semantic Search**: Uses dense vector similarity + category-based filtering.
- 🧠 **Query Classification**: Classifies questions into predefined categories like `finance`, `hr`, `marketing`, etc.
- 🤖 **LLM Answering**: Combines relevant document chunks and passes them to a Groq-hosted LLM to generate context-aware answers.
- 🖥️ **Dual UI**: 
  - `main.py` exposes a FastAPI backend for API access.
  - `app.py` provides a clean, interactive Streamlit frontend.

---

## 🗂️ Folder Structure

project-root/
│
├── services/
│ ├── classifier.py # LLM-based query classification
│ ├── search.py # Role-aware hybrid search logic
│ └── embeddings.py # Preprocessing and embedding documents
│
├── db/
│ └── clean_embeddings.pkl # Precomputed document embeddings
│
├── resources/ # Contains markdown and CSV documents
│
├── main.py # FastAPI app
├── app.py # Streamlit UI
├── .env # Environment variables (e.g., GROQ_API_KEY)
└── requirements.txt # Python dependencies




---

## 🚀 How to Run

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd <your-project-directory>



