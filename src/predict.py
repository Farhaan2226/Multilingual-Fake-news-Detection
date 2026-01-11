import torch
from transformers import AutoTokenizer
from src.model import XLMRFakeNewsClassifier

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

model = XLMRFakeNewsClassifier()
model.load_state_dict(torch.load("model.pt", map_location="cpu"))
model.eval()

def predict(text):
    enc = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=256,
        return_tensors="pt"
    )

    with torch.no_grad():
        logits = model(enc["input_ids"], enc["attention_mask"])
        pred = torch.argmax(logits, dim=1).item()

    return "FAKE ❌" if pred == 1 else "REAL ✅"


if __name__ == "__main__":
    print("🌍 Multilingual Fake News Detector")
    print("Supports English + Hindi")
    print("Type 'exit' to quit\n")

    while True:
        text = input("📰 Enter a news statement: ")

        if text.lower() == "exit":
            break

        print("Prediction:", predict(text))
        print("-" * 50)
