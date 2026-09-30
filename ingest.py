import os
import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# -----------------------------
# Configuration
# -----------------------------

PDF_PATH = "knowledge/Python_Basics_AI_Mentor_Knowledge.pdf"
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "ai_mentor_knowledge"


# -----------------------------
# Load PDF
# -----------------------------

print("Reading PDF...")

reader = PdfReader(PDF_PATH)

text = ""

for page in reader.pages:
    page_text = page.extract_text()

    if page_text:
        text += page_text + "\n"

print("PDF text extracted successfully.")


# -----------------------------
# Split text into chunks
# -----------------------------

chunks = []

chunk_size = 500

for i in range(0, len(text), chunk_size):
    chunk = text[i:i + chunk_size].strip()

    if chunk:
        chunks.append(chunk)

print(f"Created {len(chunks)} chunks.")


# -----------------------------
# Load embedding model
# -----------------------------

print("Loading embedding model...")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding model loaded.")


# -----------------------------
# Connect to ChromaDB
# -----------------------------

chroma_client = chromadb.PersistentClient(
    path=CHROMA_PATH
)


# Delete old collection if it exists
# This prevents duplicate documents
try:
    chroma_client.delete_collection(
        name=COLLECTION_NAME
    )
    print("Old knowledge collection deleted.")
except Exception:
    print("No old collection found.")


# Create fresh collection

collection = chroma_client.create_collection(
    name=COLLECTION_NAME
)


# -----------------------------
# Create embeddings
# -----------------------------

print("Creating embeddings...")

embeddings = embedding_model.encode(
    chunks
).tolist()


# -----------------------------
# Store in ChromaDB
# -----------------------------

ids = [
    f"python_chunk_{i}"
    for i in range(len(chunks))
]

collection.add(
    ids=ids,
    documents=chunks,
    embeddings=embeddings
)


# -----------------------------
# Finished
# -----------------------------

print()
print("===================================")
print("RAG INGESTION COMPLETED")
print("===================================")
print(f"PDF: {PDF_PATH}")
print(f"Chunks stored: {collection.count()}")
print(f"ChromaDB path: {CHROMA_PATH}")
print("===================================")