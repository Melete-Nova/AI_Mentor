import os
import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

KNOWLEDGE_FOLDER = "knowledge"

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to ChromaDB
chroma_client = chromadb.PersistentClient(path="chroma_db")

collection = chroma_client.get_or_create_collection(
    name="ai_mentor_knowledge"
)


def search_knowledge(question):

    # Convert question into embedding
    question_embedding = embedding_model.encode(question).tolist()

    # Search similar chunks
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=3
    )

    documents = results["documents"][0]

    context = "\n\n".join(documents)

    return context