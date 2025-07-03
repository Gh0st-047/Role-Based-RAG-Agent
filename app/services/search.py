

import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage
from getpass import getpass
from services.classifier import classify_query_with_llm
from dotenv import load_dotenv
import os



# Load environment variables
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
print(api_key)


# ──  Fake user DB 
users_db = {
    "Tony":    {"password": "password123", "role": "engineering"},
    "Bruce":   {"password": "securepass",   "role": "marketing"},
    "Sam":     {"password": "financepass",  "role": "finance"},
    "Peter":   {"password": "pete123",      "role": "engineering"},
    "Sid":     {"password": "sidpass123",   "role": "marketing"},
    "Natasha": {"password": "hrpass123",    "role": "hr"},
}

with open("db/clean_embeddings.pkl", "rb") as f:
    db = pickle.load(f)

embeddings = np.array(db["embeddings"])
texts      = db["texts"]
metadata   = db["metadata"]        # expect each item is a dict

embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ──  LLM for answer synthesis 
llm = ChatGroq(
    temperature=0,
    groq_api_key=api_key,
    model_name="meta-llama/llama-4-scout-17b-16e-instruct"
)

# ──  (Optional) quick HR‑employee‑lookup helpers  
structured_meta = []
for i, text in enumerate(texts):
    if text.startswith("Employee") and ":" in text:
        try:
            parts   = text.split(",")
            emp_id  = parts[0].split()[1].replace(":", "")
            name    = parts[0].split(":")[1].strip()
            structured_meta.append({"employee_id": emp_id, "full_name": name, "index": i})
        except Exception:
            structured_meta.append(None)
    else:
        structured_meta.append(None)


def hybrid_search(query: str, user_role: str, top_k: int = 15):
    """Return (chunks, resolved_category_list | 'unauthorized' | 'empty')."""

    predicted = classify_query_with_llm(query)
    print(f"🧠 Predicted categories: {predicted}")
    print(f"👤 User role: {user_role}")

    # HR sees everything
    if user_role == "hr":
        allowed = predicted
    else:
        allowed = list(set(predicted) & {user_role, "general"})   # intersection

    if not allowed:
        return [], "unauthorized"

    # If user tries to access HR‑sensitive personal data
    if user_role != "hr":
        q_lower = query.lower()
        if any(m and (m["employee_id"] in q_lower or m["full_name"].lower() in q_lower)
               for m in structured_meta):
            return [], "unauthorized"

    # ──  Filter embeddings by metadata 
    filtered_idx = []
    for i, meta in enumerate(metadata):
        roles_meta = None
        if isinstance(meta, dict):
            roles_meta = meta.get("allowed_roles")    # could be list or comma‑string
            if isinstance(roles_meta, str):
                roles_meta = [r.strip().lower() for r in roles_meta.split(",")]
        if roles_meta is None:   # fallback: inspect file path
            path_str   = (meta.get("path") if isinstance(meta, dict) else "") or ""
            roles_meta = [role for role in ["finance","hr","marketing","engineering","general"]
                          if role in path_str.lower()]
        if set(roles_meta) & set(allowed):
            filtered_idx.append(i)

    if not filtered_idx:
        return [], "empty"

    # ──  Rank by cosine sim  
    q_emb = embed_model.encode([query])
    sims  = cosine_similarity(q_emb, embeddings[filtered_idx])[0]
    top   = sims.argsort()[-top_k:][::-1]
    chosen_global_idx = [filtered_idx[i] for i in top]

    return [texts[i] for i in chosen_global_idx], ", ".join(allowed)


def ask_llm(question: str, context_chunks: list[str]) -> str:
    context = "\n\n".join(context_chunks)
    prompt  = (
        "You are an assistant that answers strictly from the provided company "
        "documents.\n\nContext:\n"
        f"{context}\n\nQuestion: {question}\n\nAnswer:"
    )
    return llm([HumanMessage(content=prompt)]).content

# ──  Simple login shell  
def login():
    print("🔐  Role‑based RAG — please log in.")
    username = input("Username: ").strip()
    password = getpass("Password: ")

    user = users_db.get(username)
    if user and user["password"] == password:
        print(f"✅  Logged in as {username}  (role: {user['role']})")
        return username, user["role"]
    print("❌  Invalid credentials.")
    exit()


if __name__ == "__main__":
    user, role = login()
    while True:
        q = input("\n🔍  Ask a question (or 'exit'): ").strip()
        if q.lower() == "exit":
            break

        chunks, status = hybrid_search(q, role)
        if status == "unauthorized":
            print("🚫  You are not authorized to view that. Contact HR.")
            continue
        if status == "empty" or not chunks:
            print("⚠️  No relevant information found.")
            continue

        print(f"\n📂  Using categories: {status}")
        answer = ask_llm(q, chunks)
        print("\n💬  Answer:")
        print(answer)

        print("\n📑  Sources:")
        for i, chunk in enumerate(chunks, 1):
            try:
                meta = metadata[texts.index(chunk)]
                print(f" [{i}]  {meta}")
            except ValueError:
                print(f" [{i}]  (unknown source)")
