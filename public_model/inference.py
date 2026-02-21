import torch
from backend.model import HospitalModel

model = HospitalModel()
model.load_state_dict(torch.load("latest_model.pt"))
model.eval()


def predict(x):
    x = torch.tensor(x).float()
    return model(x).item()