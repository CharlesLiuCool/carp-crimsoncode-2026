import pandas as pd

df = pd.read_csv("diabetes.csv")

# shuffle
df = df.sample(frac=1).reset_index(drop=True)

# split into two hospitals
mid = len(df) // 2

df[:mid].to_csv("hospital_A.csv", index=False)
df[mid:].to_csv("hospital_B.csv", index=False)

print("Created hospital_A.csv and hospital_B.csv")