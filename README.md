💼 Role-Based RAG Assistant
This is a Role-Based Retrieval-Augmented Generation (RAG) system built with LangChain, FastAPI, Streamlit, and Groq's LLMs. It classifies user queries, retrieves relevant document chunks based on user roles, and uses an LLM to synthesize answers strictly from those documents.

📌 Features
🔐 Role-Based Access: Ensures users can only view content relevant to their role (e.g., HR, Engineering, Finance).

🔍 Hybrid Semantic Search: Uses dense vector similarity + category-based filtering.

🧠 Query Classification: Classifies questions into predefined categories like finance, hr, marketing, etc.

🤖 LLM Answering: Combines relevant document chunks and passes them to a Groq-hosted LLM to generate context-aware answers.

🖥️ Dual UI:

main.py exposes a FastAPI backend for API access.

app.py provides a clean, interactive Streamlit frontend.



🗂️ Folder Structure

```bash
.
├── services/
│   ├── classifier.py        # Query classification using LLM
│   ├── search.py            # Role-filtered semantic search
│   └── embeddings.py        # Embeds CSV/MD documents
│
├── db/
│   └── clean_embeddings.pkl # Precomputed document embeddings
│
├── resources/               # Documents (Markdown & CSVs)
│
├── main.py                  # FastAPI backend
├── app.py                   # Streamlit UI frontend
├── .env                     # Env vars (GROQ_API_KEY)
├── requirements.txt         # Required Python libraries
└── README.md                # This file
```


🚀 How to Run
1. Clone the Repository

```bash
git clone <your-repo-url>
cd <your-project-directory>
```

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## 3. Set Up Environment Variables

Create a `.env` file in the project root:

```ini
GROQ_API_KEY=your_actual_groq_api_key_here
```

💡 Make sure you have access to Groq’s API and the correct LLaMA-4 model is supported on your key.

## 4. Generate Embeddings

Before running the app, process the documents to generate vector embeddings:

```bash
python services/embeddings.py
```

This will create `db/clean_embeddings.pkl`.

## 5. Launch the Streamlit App (Frontend)

```bash
streamlit run app.py
```

## 6. Start the FastAPI Server (Backend)

```bash
uvicorn main:app --reload
```
