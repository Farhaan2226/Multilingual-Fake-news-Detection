import pandas as pd

print("📥 Loading English training data...")
en = pd.read_csv("data/processed/train.csv")

print("📥 Loading Hindi data...")
hi = pd.read_csv("data/processed/hindi_test.csv")

# Add language column
en["language"] = "en"
hi["language"] = "hi"

print("English samples:", len(en))
print("Hindi samples:", len(hi))

# Combine
df = pd.concat([en, hi], ignore_index=True)

print("Total multilingual samples:", len(df))

# Shuffle
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Limit for CPU training
df = df.sample(n=5000, random_state=42)

print("Using for training:", len(df))

# Train / test split
train = df.iloc[:4000]
test = df.iloc[4000:]

train.to_csv("data/processed/train_multilingual.csv", index=False)
test.to_csv("data/processed/test_multilingual.csv", index=False)

print("✅ Multilingual train size:", len(train))
print("✅ Multilingual test size:", len(test))
