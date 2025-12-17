import os
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from app.model.inference import get_prediction, load_model

app = FastAPI()

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/css", StaticFiles(directory=os.path.join(STATIC_DIR, "css")), name="css")
app.mount("/js", StaticFiles(directory=os.path.join(STATIC_DIR, "js")), name="js")
app.mount("/img", StaticFiles(directory=os.path.join(STATIC_DIR, "img")), name="img")

# Templates
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Load model on startup
@app.on_event("startup")
async def startup_event():
    load_model()

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/result", response_class=HTMLResponse)
async def result(request: Request):
    return templates.TemplateResponse("result.html", {"request": request})

@app.get("/ErrorFile", response_class=HTMLResponse)
async def error_file(request: Request):
    return templates.TemplateResponse("errorDownloadFile.html", {"request": request})

@app.get("/ErrorNoFile", response_class=HTMLResponse)
async def error_no_file(request: Request):
    return templates.TemplateResponse("errorNoFile.html", {"request": request})

@app.get("/error", response_class=HTMLResponse)
async def error(request: Request):
    return templates.TemplateResponse("errorDownloadFile.html", {"request": request})

@app.get("/favicon.ico")
async def favicon():
    return RedirectResponse(url="/static/favicon.ico")

@app.post("/uploading")
async def upload_file(file: UploadFile = File(...)):
    if not file:
        return RedirectResponse(url="/ErrorNoFile", status_code=303)
    
    try:
        contents = await file.read()
        if not contents:
             return RedirectResponse(url="/ErrorNoFile", status_code=303)

        prediction = get_prediction(contents)
        
        return RedirectResponse(url=f"/result?arg={prediction}", status_code=303)
    except Exception as e:
        print(f"Error processing upload: {e}")
        return RedirectResponse(url="/ErrorFile", status_code=303)
