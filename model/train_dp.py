import sys
import os
import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from torch import nn, optim
from opacus import PrivacyEngine
from model.model import HospitalModel

# ---------------- CONFIG ----------------
CONFIG = {
    "csv_file": "hospital_client/hospital_csvs/hospital_A.csv",  # Change for hospital_B.csv
    "batch_size": 16,        # larger batch helps DP noise averaging
    "epochs": 50,            # DP slows convergence
    "lr": 1e-4,              # smaller LR under DP
    "noise_multiplier": 0.2, # smaller DP noise for tiny datasets
    "max_grad_norm": 5.0,    # allow gradients to pass signal
    "delta": 1e-5,           # for DP epsilon calculation
    "save_path": "model/private_weights.pt"
}

# Ensure repo root is on path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# ---------------- LOAD DATA ----------------
csv_path = os.path.join(repo_root, CONFIG["csv_file"])
df = pd.read_csv(csv_path)

X = torch.tensor(df.drop("Outcome", axis=1).values, dtype=torch.float32)
y = torch.tensor(df["Outcome"].values, dtype=torch.float32).unsqueeze(1)

dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=True)

# ---------------- MODEL & OPTIMIZER ----------------
# Make sure model outputs raw logits (no Sigmoid)
model = HospitalModel()
optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"])
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

# BCEWithLogitsLoss works with raw logits
criterion = nn.BCEWithLogitsLoss()

# ---------------- ATTACH DP ENGINE ----------------
privacy_engine = PrivacyEngine()
model, optimizer, loader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=loader,
    noise_multiplier=CONFIG["noise_multiplier"],
    max_grad_norm=CONFIG["max_grad_norm"]
)

# ---------------- TRAIN LOOP ----------------
for epoch in range(CONFIG["epochs"]):
    model.train()
    epoch_loss = 0.0
    n_batches = 0

    for X_batch, y_batch in loader:
        optimizer.zero_grad()
        logits = model(X_batch)  # raw logits
        loss = criterion(logits, y_batch)  # no class weight for now
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        n_batches += 1

    mean_loss = epoch_loss / max(n_batches, 1)
    scheduler.step(mean_loss)
    print(f"Epoch {epoch+1}/{CONFIG['epochs']} - Loss: {mean_loss:.4f}")

# ---------------- SAVE DP WEIGHTS ----------------
os.makedirs(os.path.dirname(CONFIG["save_path"]), exist_ok=True)
torch.save(model.state_dict(), CONFIG["save_path"])
print(f"Saved DP weights to {CONFIG['save_path']}")

# ---------------- PRINT PRIVACY SPENT ----------------
try:
    epsilon = privacy_engine.accountant.get_epsilon(delta=CONFIG["delta"])
    print(f"(ε, δ)-DP: ({epsilon:.2f}, {CONFIG['delta']})")
except Exception as e:
    print(f"Could not compute epsilon: {e}")

# ---------------- OPTIONAL: TEST INFERENCE ----------------
test_input = torch.tensor([[6,148,72,35,0,33.6,0.627,50]], dtype=torch.float32)
with torch.no_grad():
    logits = model(test_input)
    probs = torch.sigmoid(logits)
print(f"Test prediction (probability): {probs.item():.4f}")

# ---------------- DEBUG: Check raw outputs ----------------
with torch.no_grad():
    sample_logits = model(X[:5])
    print("Sample logits:", sample_logits.view(-1))
    print("Sample probabilities:", torch.sigmoid(sample_logits).view(-1))