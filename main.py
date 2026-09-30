from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from google import genai
from dotenv import load_dotenv
from rag import search_knowledge
from sentiment import analyze_sentiment
import os


# Load environment variables
load_dotenv()

# Create Gemini client
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

# Create FastAPI app
app = FastAPI()


# Serve frontend files
app.mount("/static", StaticFiles(directory="static"), name="static")


# Open the AI Mentor webpage
@app.get("/")
def home():
    return FileResponse("static/index.html")


# Request model
class ChatRequest(BaseModel):
    message: str


# AI Mentor API
@app.post("/chat")
def chat(request: ChatRequest):

    # Step 1: Analyze user's sentiment
    sentiment = analyze_sentiment(request.message)

    # Step 2: Search the RAG knowledge base
    context = search_knowledge(request.message)

    # Step 3: Create prompt using sentiment + retrieved knowledge
    prompt = f"""
    You are an AI Mentor for a college student.

    The student's current sentiment is: {sentiment}

    Adjust your tone based on the sentiment:
    - If negative, be supportive, encouraging, and patient.
    - If positive, be encouraging and motivating.
    - If neutral, be clear, friendly, and informative.

    Use the retrieved knowledge from the student's learning material
    whenever it is relevant.

    Retrieved knowledge:
    {context}

    Student's question:
    {request.message}

    Answer clearly and in a beginner-friendly way.
    """

    # Step 4: Send to Gemini
    interaction = client.interactions.create(
        model="gemini-3.6-flash",
        input=prompt
    )

    reply = interaction.output_text

    return {
        "user_message": request.message,
        "sentiment": sentiment,
        "reply": reply
    }