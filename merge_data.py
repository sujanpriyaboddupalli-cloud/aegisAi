import pandas as pd
files = [
    "data/machine-1-1.csv",
    "data/machine-1-2.csv",
    "data/machine-1-3.csv"
]

dfs = [pd.read_csv(f) for f in files]

merged = pd.concat(dfs, ignore_index=True)

print("Total rows:", len(merged))

merged.to_csv("data/final_50000.csv", index=False)