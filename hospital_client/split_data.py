import pandas as pd
import numpy as np
import json
import os

# Load data (run from project root)
csv_path = "hospital_client/diabetes.csv"
if not os.path.isfile(csv_path):
    csv_path = "diabetes.csv"
df = pd.read_csv(csv_path)

# Clean: 0 is invalid for these clinical measures; replace with median (same as Kaggle RF pipeline)
zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
for col in zero_cols:
    if col in df.columns:
        df[col] = df[col].replace(0, np.nan).fillna(df[col].median())

X = df.drop("Outcome", axis=1)
y = df["Outcome"]
feature_names = list(X.columns)

# StandardScaler-style: (x - mean) / std
mean = X.mean().values
std = X.std().values
std[std == 0] = 1.0
X_scaled = (X.values - mean) / std
X_scaled = pd.DataFrame(X_scaled, columns=feature_names)
X_scaled["Outcome"] = y.values

df = X_scaled
# Shuffle
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Split into two hospitals
mid = len(df) // 2
out_dir = "hospital_client"
os.makedirs(out_dir, exist_ok=True)
df[:mid].to_csv(os.path.join(out_dir, "hospital_A.csv"), index=False)
df[mid:].to_csv(os.path.join(out_dir, "hospital_B.csv"), index=False)

# Save scale params so inference can use same scaling (no sklearn/joblib required)
scale_params = {"mean": mean.tolist(), "std": std.tolist(), "columns": feature_names}
with open(os.path.join(out_dir, "scale_params.json"), "w") as f:
    json.dump(scale_params, f, indent=2)

print("Created hospital_A.csv and hospital_B.csv (cleaned + scaled)")
print("Saved scale_params.json to hospital_client/")