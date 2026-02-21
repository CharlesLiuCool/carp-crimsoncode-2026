import os
import requests

# ---------- CONFIG ----------
weights_path = os.path.join(os.path.dirname(__file__), "private_weights.pt")
url = "http://127.0.0.1:8000/upload"

# ---------- UPLOAD ----------
with open(weights_path, "rb") as f:
    response = requests.post(url, files={"file": f})

print(f"Uploaded weights: Status {response.status_code}, Text {response.text}")