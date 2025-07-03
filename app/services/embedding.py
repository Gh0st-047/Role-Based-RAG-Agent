import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle
import hashlib

# Config
CHUNK_SIZE = 1500
OVERLAP = 200
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OUTPUT_PATH = Path("db/clean_embeddings.pkl")

# Load embedding model
model = SentenceTransformer(MODEL_NAME)

# File paths
doc_paths = [
    Path("resources/data/engineering/engineering_master_doc.md"),
    Path("resources/data/finance/financial_summary.md"),
    Path("resources/data/finance/quarterly_financial_report.md"),
    Path("resources/data/general/employee_handbook.md"),
    Path("resources/data/hr/hr_data.csv"),
    Path("resources/data/marketing/market_report_q4_2024.md"),
    Path("resources/data/marketing/marketing_report_2024.md"),
    Path("resources/data/marketing/marketing_report_q1_2024.md"),
    Path("resources/data/marketing/marketing_report_q2_2024.md"),
    Path("resources/data/marketing/marketing_report_q3_2024.md"),
]

# Helpers
def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def read_md(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def format_employee_row(row):
    return (
        f"Employee {row['employee_id']}: {row['full_name']}, "
        f"{row['role']} in {row['department']}, "
        f"based in {row['location']}, email: {row['email']}, "
        f"born on {row['date_of_birth']}, joined on {row['date_of_joining']}, "
        f"reports to {row['manager_id']}, earns {row['salary']}, "
        f"leave balance: {row['leave_balance']}, leaves taken: {row['leaves_taken']}, "
        f"attendance: {row['attendance_pct']}%, performance rating: {row['performance_rating']} "
        f"(last reviewed on {row['last_review_date']})."
    )

def hash_chunk(text):
    return hashlib.md5(text.encode('utf-8')).hexdigest()

# Embed all chunks with deduplication
chunks = []
metadata = []
seen_hashes = set()

for path in doc_paths:
    if path.suffix == ".csv":
        df = pd.read_csv(path)
        df = df.fillna("unknown")
        required_cols = [
            "employee_id", "full_name", "role", "department", "email",
            "location", "date_of_birth", "date_of_joining", "manager_id",
            "salary", "leave_balance", "leaves_taken", "attendance_pct",
            "performance_rating", "last_review_date"
        ]
        if not all(col in df.columns for col in required_cols):
            print(f" Skipping {path}, missing required columns.")
            continue

        for _, row in df.iterrows():
            full_text = format_employee_row(row)
            for chunk in chunk_text(full_text):
                h = hash_chunk(chunk)
                if h not in seen_hashes:
                    seen_hashes.add(h)
                    chunks.append(chunk)
                    metadata.append({
                        "path": str(path),
                        "employee_id": str(row["employee_id"]).strip(),
                        "full_name": str(row["full_name"]).strip().lower()
                    })
    else:
        text = read_md(path)
        for chunk in chunk_text(text):
            h = hash_chunk(chunk)
            if h not in seen_hashes:
                seen_hashes.add(h)
                chunks.append(chunk)
                metadata.append({
                    "path": str(path)
                })

print(f"🔍 Creating embeddings for {len(chunks)} unique chunks...")
embeddings = model.encode(chunks, show_progress_bar=True)

# Ensure output directory exists
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# Save to file
with open(OUTPUT_PATH, "wb") as f:
    pickle.dump({
        "embeddings": embeddings,
        "texts": chunks,
        "metadata": metadata
    }, f)

print(f"Saved {len(embeddings)} unique chunk embeddings to {OUTPUT_PATH}")
