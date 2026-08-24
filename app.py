from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from rag import ask_question


# Get the folder where app.py exists
BASE_DIR = Path(__file__).resolve().parent

STATIC_DIR = BASE_DIR / "static"
print("APP LOCATION:", BASE_DIR)
print("STATIC LOCATION:", STATIC_DIR)
print("SCRIPT EXISTS:", (STATIC_DIR / "script.js").exists())
TEMPLATES_DIR = BASE_DIR / "templates"


app = FastAPI(title="AgriAI")


# Serve CSS, JS and images
app.mount(
    "/static",
    StaticFiles(directory=str(STATIC_DIR)),
    name="static"
)


# Serve HTML
templates = Jinja2Templates(
    directory=str(TEMPLATES_DIR)
)


class ChatRequest(BaseModel):
    message: str


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )


@app.post("/chat")
async def chat(request: ChatRequest):

    answer = ask_question(request.message)

    return {
        "answer": answer
    }