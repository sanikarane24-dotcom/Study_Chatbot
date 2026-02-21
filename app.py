import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pymongo import MongoClient
from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Load environment variables
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
mongo_uri = os.getenv("MONGODB_URI")

# MongoDB Connection
client = MongoClient(mongo_uri)
db = client["Chatbot"]
collection = db["users"]

app=FastAPI()

class ChatRequest(BaseModel):
    user_id:str
    question:str

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True
)

# Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a knowledgeable assistant that helps students with their questions. Use the conversation history to provide context-aware responses.Also make the response pretty, alligned and well structured."),
        ("placeholder","{history}"),
        ("user", "{question}")
    ]
)

# LLM Model
chat = ChatGroq(
    api_key=groq_api_key,
    model="openai/gpt-oss-20b"
)

chain = prompt | chat



def get_user_history(user_id):
    chats=collection.find({"user_id":user_id}).sort("timestamp",1)
    history=[]

    for chat in chats:
        history.append((chat["role"],chat["message"]))
    return history

@app.get("/")
def home():
    return {"message":"Welcome to the Study Chatbot!"}

@app.post("/chat")
def chat(request:ChatRequest):
    history=get_user_history(request.user_id)
    response = chain.invoke({"history":history, "question": request.question})
     
    collection.insert_one({
        "user_id": request.user_id,
        "role": "user",
        "message": request.question,
        "timestamp": datetime.utcnow()
    })

    # Store assistant response
    collection.insert_one({
        "user_id": request.user_id,
        "role": "assistant",
        "message": response.content,
        "timestamp": datetime.utcnow()
    })
     
    return {"response":response.content}




# # Chat Loop
# while True:
#     question = input("Ask a question: ")

#     if question.lower() in {"exit", "quit"}:
#         print("Chat ended.")
#         break
    
#     history=get_user_history(user_id)
#     # Generate response
#     response = chain.invoke({"history":history, "question": question})

#     # Store user message
#     collection.insert_one({
#         "user_id": user_id,
#         "role": "user",
#         "message": question,
#         "timestamp": datetime.utcnow()
#     })

#     # Store assistant response
#     collection.insert_one({
#         "user_id": user_id,
#         "role": "assistant",
#         "message": response.content,
#         "timestamp": datetime.utcnow()
#     })

#     # Print response
#     print("\nAssistant:", response.content, "\n")

    
