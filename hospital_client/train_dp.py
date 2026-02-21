import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from opacus import PrivacyEngine
from backend.model import HospitalModel

# Load hospital data
csv = pd.read_csv("hospital_client/hospital_A.csv")
X = torch.tensor(csv.iloc[:, :-1].values).float()
y = torch.tensor(csv.iloc[:, -1].values).float().unsqueeze(1)

dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = HospitalModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

privacy_engine = PrivacyEngine()
model, optimizer, loader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=loader,
    noise_multiplier=1.0,
    max_grad_norm=1.0,
)

# Train
for epoch in range(5):
    for X_batch, y_batch in loader:
        pred = model(X_batch)
        loss = ((pred - y_batch) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Save private weights
torch.save(model.state_dict(), "private_weights.pt")

print("Saved DP weights")