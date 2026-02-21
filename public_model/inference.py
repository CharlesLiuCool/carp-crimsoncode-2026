import torch
from backend.model import HospitalModel

model = HospitalModel()
model.load_state_dict(torch.load("latest_model.pt"))
model.eval()


def predict(x):
    """Return probability of having diabetes as a percentage (0-100)."""
    x = torch.tensor(x).float()
    prob = model(x).item()
    return round(prob * 100, 2)