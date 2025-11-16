from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.agent import build_agent
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title= "Open Source Chatbot API")

# Build and compile the agent graph once at startup
agent_graph = build_agent().compile()

origins = [
    "http://localhost:8000",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        # Invoke the agent with the user's message
        result = agent_graph.invoke({
            "messages": [{"role": "user", "content": request.message}]
        })
        
        # Extract the final response
        final_message = result["messages"][-1]
        response_content = final_message.content if hasattr(final_message, 'content') else str(final_message)
        
        return {"response": response_content}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



