from fastapi import FastAPI
import torch
from backend.model import HospitalModel

app = FastAPI()
public_model_path = "backend/public_model/latest_model.pt"
model = HospitalModel()
model.load_state_dict(torch.load(public_model_path))
model.eval()

@app.get("/predict")
def predict():
    # example dummy input
    test_input = torch.tensor([[6,148,72,35,0,33.6,0.627,50]], dtype=torch.float32)
    prediction = model(test_input)
    return {"prediction": prediction.item()}