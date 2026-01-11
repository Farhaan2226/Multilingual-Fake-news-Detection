import os
import torch
import streamlit as st
from transformers import AutoTokenizer
from src.model import XLMRFakeNewsClassifier

# -------------------------
# Page config
# -------------------------
st.set_page_config(
    page_title="Multilingual Fake News Detector",
    layout="centered"
)

st.title("🌍 Multilingual Fake News Detector")
st.markdown("Supports **English & Hindi** using XLM-RoBERTa")

# -------------------------
# Load model & tokenizer
# -------------------------
@st.cache_resource
def load_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, "model.pt")

    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

    model = XLMRFakeNewsClassifier()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    return tokenizer, model


tokenizer, model = load_model()

# -------------------------
# UI
# -------------------------
text = st.text_area(
    "Enter a news statement (English or Hindi)",
    height=150,
    placeholder="भारत सरकार ने सभी सोशल मीडिया प्लेटफॉर्म बंद कर दिए हैं..."
)

# -------------------------
# Prediction
# -------------------------
if st.button("Analyze"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        enc = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=256,
            return_tensors="pt"
        )

        with torch.no_grad():
            logits = model(enc["input_ids"], enc["attention_mask"])
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred].item()

        if pred == 1:
            st.error(f"🚨 **FAKE NEWS**  \nConfidence: {confidence:.2f}")
        else:
            st.success(f"✅ **REAL NEWS**  \nConfidence: {confidence:.2f}")

# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.caption("Model: XLM-RoBERTa fine-tuned on English + Hindi fake-news datasets")
