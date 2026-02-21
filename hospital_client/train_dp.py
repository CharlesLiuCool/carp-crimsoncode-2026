import os
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from opacus import PrivacyEngine
from backend.model import HospitalModel

# ---------- CONFIG ----------
HOSPITAL_CSV = "hospital_A.csv"  # change to hospital_B.csv for other hospital
BATCH_SIZE = 32
EPOCHS = 5
LR = 0.01
NOISE_MULTIPLIER = 1.0  # adjust for privacy
MAX_GRAD_NORM = 1.0
DELTA = 1e-5

# ---------- LOAD DATA ----------
df = pd.read_csv(os.path.join(os.path.dirname(__file__), HOSPITAL_CSV))
X = torch.tensor(df.drop("Outcome", axis=1).values, dtype=torch.float32)
y = torch.tensor(df["Outcome"].values, dtype=torch.float32).unsqueeze(1)

dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ---------- INIT MODEL ----------
model = HospitalModel()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr=LR)

# ---------- ATTACH DP ----------
privacy_engine = PrivacyEngine(
    model,
    batch_size=BATCH_SIZE,
    sample_size=len(dataset),
    alphas=[10, 100],
    noise_multiplier=NOISE_MULTIPLIER,
    max_grad_norm=MAX_GRAD_NORM
)
privacy_engine.attach(optimizer)

# ---------- TRAIN ----------
for epoch in range(EPOCHS):
    for xb, yb in dataloader:
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# ---------- SAVE DP WEIGHTS ----------
os.makedirs("hospital_client", exist_ok=True)
weights_path = os.path.join(os.path.dirname(__file__), "private_weights.pt")
torch.save(model.state_dict(), weights_path)
print(f"Saved DP weights to {weights_path}")

# ---------- PRINT PRIVACY SPENT ----------
epsilon, best_alpha = privacy_engine.get_privacy_spent(delta=DELTA)
print(f"(ε, δ)-DP: ({epsilon:.2f}, {DELTA})")