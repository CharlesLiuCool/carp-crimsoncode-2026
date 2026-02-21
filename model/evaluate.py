import sys
import os

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, repo_root)

import torch
import pandas as pd
from model.model import HospitalModel

weights_path = os.path.join(os.path.dirname(__file__), "private_weights.pt")
if not os.path.isfile(weights_path):
    print("Run model/train_dp.py first to create model/private_weights.pt")
    sys.exit(1)

model = HospitalModel()
state = torch.load(weights_path)
# Opacus wraps the model, so saved keys may have "_module." prefix
if any(k.startswith("_module.") for k in state):
    state = {k.replace("_module.", ""): v for k, v in state.items()}
model.load_state_dict(state)
model.eval()

data_path = os.path.join(repo_root, "hospital_client/hospital_B.csv")
csv = pd.read_csv(data_path)
X_test = torch.tensor(csv.iloc[:, :-1].values).float()
y_test = torch.tensor(csv.iloc[:, -1].values).float().unsqueeze(1)

with torch.no_grad():
    pred = model(X_test)
pred_class = (pred >= 0.5).float()

accuracy = (pred_class == y_test).float().mean().item()

# Precision, recall, F1 for positive class (diabetes)
tp = ((pred_class == 1) & (y_test == 1)).sum().item()
fp = ((pred_class == 1) & (y_test == 0)).sum().item()
fn = ((pred_class == 0) & (y_test == 1)).sum().item()

precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

print("Test set (hospital_B.csv) metrics:")
print(f"  Accuracy:  {accuracy:.4f}")
print(f"  Precision (diabetes): {precision:.4f}")
print(f"  Recall (diabetes):    {recall:.4f}")
print(f"  F1 (diabetes):        {f1:.4f}")
