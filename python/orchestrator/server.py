import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

def load_dotenv():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    paths = [
        os.path.join(current_dir, "..", "..", ".env"),
        os.path.join(current_dir, "..", ".env"),
        os.path.join(os.getcwd(), ".env"),
    ]
    for path in paths:
        if os.path.exists(path):
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split("=", 1)
                    if len(parts) == 2:
                        key = parts[0].strip()
                        val = parts[1].strip()
                        if val.startswith(('"', "'")) and val.endswith(('"', "'")):
                            val = val[1:-1]
                        os.environ[key] = val
            break

load_dotenv()

from .main import ChatSession

app = FastAPI(title="MiniBrain Chatbot Interface")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

session = ChatSession()


class ChatRequest(BaseModel):
    message: str


@app.post("/api/chat")
async def chat(req: ChatRequest):
    result = session.handle_structured(req.message)
    return result


@app.post("/api/reset")
async def reset():
    session.state.reset()
    return {"status": "success", "message": "Dialogue state and C++ LSTM cell memory reset successfully."}


# Mount static files directory
static_dir = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir)

app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

if __name__ == "__main__":
    uvicorn.run("orchestrator.server:app", host="127.0.0.1", port=8000, reload=True)
