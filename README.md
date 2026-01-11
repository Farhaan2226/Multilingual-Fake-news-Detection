This project implements a multilingual fake news detection system capable of classifying news articles written in multiple Indian languages (Hindi, Tamil, Malayalam, Gujarati, etc.) using a cross-lingual NLP model trained on English data.

The system demonstrates how a model trained in one language can generalize to other languages using multilingual transformer embeddings.

It includes:

       Data processing pipelines

       Model training

       Cross-language evaluation

       A live Streamlit web app

We use a multilingual transformer-based classifier trained on English fake-news data and evaluated on Hindi and other Indian languages.

Pipeline:

Text → Multilingual Transformer → Sentence Embeddings → Classifier → Fake / Real


The transformer maps all languages into a shared semantic space, allowing zero-shot transfer.

Datasets Used
English (Training)

Fake and real news articles from standard English datasets

Indian Languages (Testing)

BharatFakeNewsKosh

Hindi

Tamil

Malayalam

Gujarati

Bangla

Odia

Assamese

Telugu

These datasets contain verified misinformation collected from Indian fact-checking organizations.

We perform zero-shot cross-lingual evaluation:

Language	Samples	    Accuracy	Macro-F1
Hindi	    4,192	       ~0.58	   ~0.36

The performance drop compared to English highlights real-world domain shift, making this a realistic research-grade system.
A user-friendly web app allows anyone to test the model:

Features:

Input news in any language

Instant fake / real classification

Works across English and Indian languages

Run locally:

streamlit run app.py

Multilingual-Fake-News-Detection/
├── src/                -> Training, evaluation, model code
├── data/               -> Data processing scripts
├── notebooks/          -> Analysis & experiments
├── experiments/        -> Logs & results
├── app.py              -> Streamlit web app
├── requirements.txt
└── README.md

Key NLP Concepts Used :

Multilingual Transformers

Cross-lingual Transfer Learning

Zero-shot Learning

Text Embeddings

Fake News Classification

Domain Shift

