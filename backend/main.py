from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
import torch
from backend.aggregator import aggregate
from backend.model import HospitalModel

# -------------------- APP SETUP --------------------
app = FastAPI()

# CORS: allow your frontend to call the API
origins = [
    "http://127.0.0.1:8081",
    "http://localhost:8081"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,       # or ["*"] for all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- UPLOAD FOLDER --------------------
UPLOAD_FOLDER = "backend/temp"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -------------------- UPLOAD AND AGGREGATE --------------------
@app.post("/upload")
async def upload_weights(file: UploadFile = File(...)):
    try:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Aggregate all DP weights into latest public model
        aggregate()

        return {"status": "aggregated"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}

# -------------------- PREDICTION ENDPOINT --------------------
@app.get("/predict")
def predict():
    public_model_path = "backend/public_model/latest_model.pt"

    if not os.path.exists(public_model_path):
        return {"status": "error", "detail": "No public model found."}

    try:
        model = HospitalModel()
        model.load_state_dict(torch.load(public_model_path))
        model.eval()

        # Example test input
        test_input = torch.tensor([[6,148,72,35,0,33.6,0.627,50]], dtype=torch.float32)
        prediction = model(test_input)

        return {"prediction": prediction.item()}
    except Exception as e:
        return {"status": "error", "detail": str(e)}