import torch
import pandas as pd
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score

from src.model import XLMRFakeNewsClassifier


def evaluate_model(model, texts, labels, tokenizer):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    preds = []

    with torch.no_grad():
        for text in texts:
            encoding = tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=256,
                return_tensors="pt"
            )

            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)

            logits = model(input_ids, attention_mask)
            pred = torch.argmax(logits, dim=1).item()
            preds.append(pred)

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")

    return acc, f1


if __name__ == "__main__":
    # Load test data
    df = pd.read_csv("data/processed/test.csv")

    texts = df["text"].tolist()
    labels = df["label"].tolist()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

    # Load trained model
    model = XLMRFakeNewsClassifier()
    model.load_state_dict(torch.load("model.pt", map_location="cpu"))

    acc, f1 = evaluate_model(model, texts, labels, tokenizer)

    print("📊 Evaluation Results")
    print("Accuracy:", round(acc, 4))
    print("Macro F1:", round(f1, 4))
