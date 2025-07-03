
import os
import streamlit as st
import pickle
import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage
from services.classifier import classify_query_with_llm
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")



with open("db/clean_embeddings.pkl", "rb") as f:
    db = pickle.load(f)

embeddings = np.array(db["embeddings"])
texts      = db["texts"]
metadata   = db["metadata"]

import torch

# Fix for meta tensor error
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU usage


embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
llm = ChatGroq(
    temperature=0,
    groq_api_key=api_key,
    model_name="meta-llama/llama-4-scout-17b-16e-instruct"
)

# Fake users
users_db = {
    "Tony":    {"password": "password123", "role": "engineering"},
    "Bruce":   {"password": "securepass",   "role": "marketing"},
    "Sam":     {"password": "financepass",  "role": "finance"},
    "Peter":   {"password": "pete123",      "role": "engineering"},
    "Sid":     {"password": "sidpass123",   "role": "marketing"},
    "Natasha": {"password": "hrpass123",    "role": "hr"},
}


structured_meta = []
for i, text in enumerate(texts):
    if text.startswith("Employee") and ":" in text:
        try:
            emp_id  = text.split()[1].replace(":", "")
            name    = text.split(":")[1].split(",")[0].strip()
            structured_meta.append({"employee_id": emp_id, "full_name": name, "index": i})
        except Exception:
            structured_meta.append(None)
    else:
        structured_meta.append(None)

def hybrid_search(query: str, user_role: str, top_k: int = 20):
    predicted = classify_query_with_llm(query)        
    st.session_state.debug_predicted = predicted      

    if user_role == "hr":
        allowed = predicted
    else:
        allowed = list(set(predicted) & {user_role, "general"})

    if not allowed:
        return [], "unauthorized"

    
    q_lower   = query.lower()
    emp_id    = next((t for t in q_lower.split() if t.isdigit()), None)
    name_hit  = next((m["full_name"] for m in structured_meta
                      if m and m["full_name"].lower() in q_lower), None)
    if (emp_id or name_hit) and user_role != "hr":
        return [], "unauthorized"

    # --- metadata role filtering ---
    filtered_idx = []
    for i, meta in enumerate(metadata):
        roles_meta = None
        if isinstance(meta, dict):
            roles_meta = meta.get("allowed_roles")
            if isinstance(roles_meta, str):
                roles_meta = [r.strip().lower() for r in roles_meta.split(",")]
        if roles_meta is None:   # fallback to path sniff
            path_str   = (meta.get("path") if isinstance(meta, dict) else "") or ""
            roles_meta = [r for r in ["finance","hr","marketing","engineering","general"]
                          if r in path_str.lower()]
        if set(roles_meta) & set(allowed):
            filtered_idx.append(i)

    if not filtered_idx:
        return [], "empty"

    q_emb   = embed_model.encode([query])
    sims    = cosine_similarity(q_emb, embeddings[filtered_idx])[0]
    top_loc = sims.argsort()[-top_k:][::-1]
    glob_idx = [filtered_idx[i] for i in top_loc]

    return [texts[i] for i in glob_idx], ", ".join(allowed)

# ──  LLM question‑answering  
def ask_llm(question: str, context_chunks: List[str]) -> str:
    context = "\n\n".join(context_chunks)
    prompt  = (
        "You are an assistant that answers strictly from the provided company "
        "documents.\n\nContext:\n"
        f"{context}\n\nQuestion: {question}\n\nAnswer:"
    )
    return llm([HumanMessage(content=prompt)]).content

# ──  Streamlit UI  
st.set_page_config(page_title="Role‑Based RAG Assistant", layout="wide")
st.title("💼 Role‑Based RAG Assistant")

if "auth" not in st.session_state:
    st.session_state.auth = False

if not st.session_state.auth:
    with st.form(key="login"):
        st.subheader("🔐 Login")
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        if st.form_submit_button("Login"):
            user = users_db.get(u)
            if user and user["password"] == p:
                st.session_state.update({"auth": True, "username": u, "role": user["role"]})
                st.success(f"Welcome {u}!  (role: {user['role']})")
            else:
                st.error("❌  Invalid credentials")
    st.stop()

query = st.text_input("💬 Ask a question")
if st.button("🔍 Search") and query:
    with st.spinner("Retrieving context…"):
        chunks, status = hybrid_search(query, st.session_state.role)

    if status == "unauthorized":
        st.error("🚫  You are not authorized to view that information.")
    elif status == "empty" or not chunks:
        st.warning("⚠️  No relevant information found.")
    else:
        st.success(f"🔖 Categories used: `{status}`")
        answer = ask_llm(query, chunks)
        st.markdown("### 💡 Answer")
        st.markdown(answer)

        st.markdown("### 📚 References")
        for i, chunk in enumerate(chunks, 1):
            idx  = texts.index(chunk)
            meta = metadata[idx]
            with st.expander(f"Reference #{i}"):
                st.code(chunk[:600] + ("…" if len(chunk) > 600 else ""), language="text")
                st.json(meta)


st.sidebar.title("User Panel")
st.sidebar.markdown(f"**User:** {st.session_state.username}")
st.sidebar.markdown(f"**Role:** {st.session_state.role}")
if st.sidebar.checkbox("Show debug (predicted categories)"):
    st.sidebar.write(st.session_state.get("debug_predicted", []))
if st.sidebar.button("Logout"):
    for k in ["auth", "username", "role"]:
        st.session_state.pop(k, None)
    st.experimental_rerun()
