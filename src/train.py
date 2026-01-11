import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
import pandas as pd

from src.dataset import FakeNewsDataset
from src.model import XLMRFakeNewsClassifier


def train_model(train_texts, train_labels, epochs=3, batch_size=4):
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

    dataset = FakeNewsDataset(train_texts, train_labels, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = XLMRFakeNewsClassifier()
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    loss_fn = torch.nn.CrossEntropyLoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(epochs):
        total_loss = 0.0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, attention_mask)
            loss = loss_fn(logits, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} | Avg Loss: {avg_loss:.4f}")

    return model


if __name__ == "__main__":
    df = pd.read_csv("data/processed/train_multilingual.csv")


    texts = df["text"].tolist()
    labels = df["label"].tolist()

    model = train_model(texts, labels, epochs=1)
    torch.save(model.state_dict(), "model.pt")
