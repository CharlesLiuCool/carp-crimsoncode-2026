
import os

import numpy as np
import pandas as pd

# Load data (run from project root)
csv_path = "hospital_client/diabetes.csv"
if not os.path.isfile(csv_path):
    csv_path = "diabetes.csv"
if not os.path.isfile(csv_path):
    csv_path = "hospital_client/merged_diabetes.csv"
if not os.path.isfile(csv_path):
    raise FileNotFoundError(
        "No input CSV found. Place diabetes.csv or merged_diabetes.csv in hospital_client/ or project root."
    )
df = pd.read_csv(csv_path)
outcome_col = "Outcome"
if outcome_col not in df.columns:
    raise ValueError("CSV must contain an 'Outcome' column.")
has_source = "Source" in df.columns

# Clean: 0 is invalid for these clinical measures; replace with median
zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
for col in zero_cols:
    if col in df.columns:
        df[col] = df[col].replace(0, np.nan).fillna(df[col].median())

# Keep feature columns + Outcome + Source (if present). No scaling.
feature_cols = [c for c in df.columns if c != outcome_col and c != "Source"]
df = df[feature_cols + [outcome_col] + (["Source"] if has_source else [])].copy()

# Shuffle
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Split into small, medium, and large datasets (20% / 35% / 45%)
n = len(df)
i_small = int(n * 0.20)
i_medium = i_small + int(n * 0.35)
# small: [0 : i_small], medium: [i_small : i_medium], large: [i_medium : n]
out_dir = "hospital_client"
os.makedirs(out_dir, exist_ok=True)
df[:i_small].to_csv(os.path.join(out_dir, "hospital_small.csv"), index=False)
df[i_small:i_medium].to_csv(os.path.join(out_dir, "hospital_medium.csv"), index=False)
df[i_medium:].to_csv(os.path.join(out_dir, "hospital_large.csv"), index=False)

print("Created hospital_small.csv, hospital_medium.csv, hospital_large.csv (cleaned, raw values)")
print("Use build_scaler_from_merged(merged_diabetes.csv) and train.py to scale at training time.")
