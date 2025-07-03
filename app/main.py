from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict

from services.search import hybrid_search, ask_llm
from services.classifier import classify_query_with_llm

app = FastAPI(title="Role-Based RAG Agent")

#  Dummy users DB with roles
users_db: Dict[str, Dict[str, str]] = {
    "Tony": {"password": "password123", "role": "engineering"},
    "Bruce": {"password": "securepass", "role": "marketing"},
    "Sam": {"password": "financepass", "role": "finance"},
    "Peter": {"password": "pete123", "role": "engineering"},
    "Sid": {"password": "sidpass123", "role": "marketing"},
    "Natasha": {"password": "hrpass123", "role": "hr"}
}

#  Auth setup
security = HTTPBasic()

def authenticate(credentials: HTTPBasicCredentials = Depends(security)) -> Dict:
    user = users_db.get(credentials.username)
    if not user or user["password"] != credentials.password:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    return {"username": credentials.username, "role": user["role"]}

# Allow CORS if using a frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def health_check():
    return {"status": "ok", "message": "RAG API is live."}

@app.post("/chat")
def chat(message: str, user: Dict = Depends(authenticate)):
    username = user["username"]
    role = user["role"]

    # 1. Classify the query
    category = classify_query_with_llm(message)

    # 2. Search role-relevant content
    chunks, status = hybrid_search(message, role)

    if status == "unauthorized":
        raise HTTPException(status_code=403, detail="Access denied based on your role.")
    if not chunks:
        return {
            "username": username,
            "role": role,
            "category": category,
            "answer": "No relevant documents found.",
            "sources": []
        }

    # 3. Get LLM-based answer
    answer = ask_llm(message, chunks)

    return {
        "username": username,
        "role": role,
        "category": category,
        "answer": answer,
        "sources": chunks
    }
