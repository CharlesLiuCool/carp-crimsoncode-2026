from fastapi import FastAPI, UploadFile, File
import os
import shutil
from backend.aggregator import aggregate

app = FastAPI()

UPLOAD_FOLDER = "backend/temp"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.post("/upload")
async def upload_weights(file: UploadFile = File(...)):
    try:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Aggregate all DP weights
        aggregate()

        return {"status": "aggregated"}
    except Exception as e:
        # Return the exception as JSON for debugging
        return {"status": "error", "detail": str(e)}