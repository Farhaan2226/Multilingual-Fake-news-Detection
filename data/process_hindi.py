import pandas as pd

RAW = "data/raw/bharatfakenewskosh.xlsx"
OUT = "data/processed/hindi_test.csv"

print("📥 Loading BharatFakeNewsKosh...")
df = pd.read_excel(RAW)

# Clean language column
df["Language"] = df["Language"].astype(str).str.strip()

# Keep Hindi only
df = df[df["Language"] == "Hindi"]

# Select correct columns
df = df[["Text", "Label"]]

# Rename
df = df.rename(columns={
    "Text": "text",
    "Label": "label"
})

# Normalize labels
df["label"] = df["label"].astype(str).str.strip()

# Map labels
df["label"] = df["label"].map({
    "False": 1,   # Fake
    "True": 0    # Real
})

# Drop invalid rows
df = df[df["label"].isin([0, 1])]
df = df.dropna()

df.to_csv(OUT, index=False)

print("✅ Hindi test set created:", OUT)
print("📊 Hindi samples:", len(df))
