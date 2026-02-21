from fastapi import FastAPI, UploadFile
import shutil
import os
from backend.aggregator import aggregate

app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.post("/upload")
async def upload(file: UploadFile):

    path = os.path.join(UPLOAD_DIR, file.filename)

    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    aggregate()

    return {"status": "aggregated"}


@app.get("/health")
def health():
    return {"status": "ok"}