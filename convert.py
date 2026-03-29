import os
import pandas as pd
import numpy as np

# ✅ INPUT FILES (ALL 3)
INPUT_FILES = [
    "data/machine-1-1.txt",
    "data/machine-1-2.txt",
    "data/machine-1-3.txt"
]

# ✅ OUTPUT FILES
OUTPUT_FILES = [
    "data/machine-1-1.csv",
    "data/machine-1-2.csv",
    "data/machine-1-3.csv"
]

# Column names
SMD_COLUMN_NAMES = [
    "cpu","net_in","net_out","network","temp","power",
    "mem","disk_read","disk_write","load_1","load_5","load_15"
]

for i in range(12, 38):
    SMD_COLUMN_NAMES.append(f"feat_{i:02d}")

TEMP_MIN, TEMP_MAX = 30.0, 100.0


def scale_column(series, name):
    if name == "temp":
        return series * (TEMP_MAX - TEMP_MIN) + TEMP_MIN
    elif name in ["cpu", "power", "network", "mem"]:
        return series * 100
    return series


def convert_file(input_path, output_path):
    print(f"\n📂 Processing: {input_path}")

    if not os.path.exists(input_path):
        print(f"❌ File not found: {input_path}")
        return

    df = pd.read_csv(input_path, header=None, names=SMD_COLUMN_NAMES)

    # Add timestamp
    df.insert(
        0,
        "timestamp",
        pd.date_range(start="2024-01-01", periods=len(df), freq="1min")
    )

    # Scale important columns
    for col in ["cpu", "temp", "power", "network", "mem"]:
        df[col] = scale_column(df[col], col)

    # Save CSV
    df.to_csv(output_path, index=False)
    print(f"✅ Saved: {output_path} ({len(df)} rows)")


# 🔄 Convert all files
for inp, out in zip(INPUT_FILES, OUTPUT_FILES):
    convert_file(inp, out)


# 🚀 MERGE ALL FILES INTO ONE BIG DATASET
print("\n🔗 Merging all datasets...")

dfs = []
for file in OUTPUT_FILES:
    if os.path.exists(file):
        dfs.append(pd.read_csv(file))

merged = pd.concat(dfs, ignore_index=True)

# Save final dataset
merged.to_csv("data/final_60000.csv", index=False)

print(f"✅ Final merged dataset saved: data/final_60000.csv")
print(f"📊 Total rows: {len(merged)}")