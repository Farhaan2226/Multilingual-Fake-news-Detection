import torch
import pandas as pd
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score

from src.model import XLMRFakeNewsClassifier


# Load Hindi test data
df = pd.read_csv("data/processed/hindi_test.csv")

texts = df["text"].tolist()
labels = df["label"].tolist()

print("📊 Hindi samples:", len(texts))

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

# Load trained model
model = XLMRFakeNewsClassifier()
model.load_state_dict(torch.load("model.pt", map_location="cpu"))
model.eval()

preds = []

with torch.no_grad():
    for text in texts:
        enc = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=256,
            return_tensors="pt"
        )

        logits = model(enc["input_ids"], enc["attention_mask"])
        preds.append(torch.argmax(logits, dim=1).item())

print("📊 Hindi Zero-Shot Results")
print("Accuracy:", round(accuracy_score(labels, preds), 4))
print("Macro F1:", round(f1_score(labels, preds, average="macro"), 4))
