import pandas as pd
import os

# ======================
# PATHS
# ======================
RAW_DIR = "data/raw"
OUT_DIR = "data/processed"

print("🚀 Starting dataset processing...")
print("📂 Raw files found:", os.listdir(RAW_DIR))

os.makedirs(OUT_DIR, exist_ok=True)

# ======================
# 1. LOAD ENGLISH DATA (ISOT)
# ======================

true_path = os.path.join(RAW_DIR, "True.csv")
fake_path = os.path.join(RAW_DIR, "Fake.csv")

print("📄 Reading:", true_path)
true_df = pd.read_csv(true_path)

print("📄 Reading:", fake_path)
fake_df = pd.read_csv(fake_path)

print("✅ True rows:", len(true_df))
print("✅ Fake rows:", len(fake_df))

# ======================
# 2. LABEL & LANGUAGE
# ======================

true_df["label"] = 0   # Real
fake_df["label"] = 1   # Fake

true_df["language"] = "en"
fake_df["language"] = "en"

# ======================
# 3. TEXT CREATION
# ======================
# Use title + body for stronger signal

true_df["text"] = true_df["title"].astype(str) + ". " + true_df["text"].astype(str)
fake_df["text"] = fake_df["title"].astype(str) + ". " + fake_df["text"].astype(str)

true_df = true_df[["text", "label", "language"]]
fake_df = fake_df[["text", "label", "language"]]

# ======================
# 4. COMBINE
# ======================

df = pd.concat([true_df, fake_df], ignore_index=True)
print("📊 Total samples before sampling:", len(df))

# ======================
# 5. LIMIT DATASET SIZE (FAST TRAINING)
# ======================

df = df.sample(n=2000, random_state=42)
print("⚡ Samples after limiting:", len(df))

# ======================
# 6. SHUFFLE
# ======================

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# ======================
# 7. TRAIN / TEST SPLIT
# ======================

split_idx = int(0.8 * len(df))
train_df = df.iloc[:split_idx]
test_df = df.iloc[split_idx:]

train_path = os.path.join(OUT_DIR, "train.csv")
test_path = os.path.join(OUT_DIR, "test.csv")

train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)

print("📁 Saved files:")
print("   ➤", train_path)
print("   ➤", test_path)

print("📊 Train size:", len(train_df))
print("📊 Test size:", len(test_df))

print("🎉 Dataset processing completed successfully.")
