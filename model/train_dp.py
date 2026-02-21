import sys
import os

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, repo_root)

import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from opacus import PrivacyEngine
from model.model import HospitalModel

# Load hospital data (paths relative to repo root)
data_path = os.path.join(repo_root, "hospital_client/hospital_A.csv")
csv = pd.read_csv(data_path)
X = torch.tensor(csv.iloc[:, :-1].values).float()
y = torch.tensor(csv.iloc[:, -1].values).float().unsqueeze(1)

# Class balance: weight positive class (minority) so loss treats classes more equally
n_pos = (y == 1).sum().item()
n_neg = (y == 0).sum().item()
pos_weight_val = n_neg / max(n_pos, 1)
criterion = torch.nn.BCELoss(reduction="none")

dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = HospitalModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=3
)

privacy_engine = PrivacyEngine()
model, optimizer, loader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=loader,
    noise_multiplier=1.0,
    max_grad_norm=1.0,
)

# Train with BCE and class weighting; more epochs + LR scheduler for better convergence
for epoch in range(60):
    model.train()
    epoch_loss = 0.0
    n_batches = 0
    for X_batch, y_batch in loader:
        pred = model(X_batch)
        bce = criterion(pred, y_batch)
        w = (y_batch == 1).float() * (pos_weight_val - 1) + 1
        loss = (bce * w).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        n_batches += 1
    mean_loss = epoch_loss / max(n_batches, 1)
    scheduler.step(mean_loss)

# Save private weights to model folder
weights_path = os.path.join(os.path.dirname(__file__), "private_weights.pt")
torch.save(model.state_dict(), weights_path)

print("Saved DP weights to model/private_weights.pt")
