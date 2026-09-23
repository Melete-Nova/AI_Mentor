from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class ChatRequest(BaseModel):
    message: str


@app.get("/")
def home():
    return {"message": "Welcome to AI Mentor"}


@app.post("/chat")
def chat(request: ChatRequest):

    user_message = request.message.lower()

    if "hello" in user_message or "hi" in user_message:
        reply = "Hello! I am your AI Mentor. How can I help you?"

    elif "python" in user_message:
        reply = "Python is a popular programming language used for web development, data science, and AI."

    elif "machine learning" in user_message:
        reply = "Machine Learning allows computers to learn patterns from data and make predictions."

    else:
        reply = "I am your AI Mentor. Please ask me a question about your studies or career."

    return {
        "user_message": request.message,
        "reply": reply
    }