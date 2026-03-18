# ===============================
# Streamlit Corporate Demography App
# ===============================

import streamlit as st
import pandas as pd
import numpy as np
import torch
import re

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForTokenClassification
)

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(
    page_title="Corporate Demography Analytics",
    layout="wide"
)

st.title("📊 Corporate Demography Analytics Dashboard")
st.write("LLM-based extraction and semantic analysis of corporate demographic information")

# -------------------------------
# Load dataset
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("synthetic_corporate_dataset_full.csv")
    return df

df = load_data()

# -------------------------------
# Create Combined_Text (CRITICAL FIX)
# -------------------------------
df["Combined_Text"] = (
    df["Business_Description"].fillna("") + " " +
    df["Ownership_Declaration"].fillna("") + " " +
    df["Narrative_Statement"].fillna("")
)

# -------------------------------
# Load models
# -------------------------------
@st.cache_resource
def load_models():
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    ner_model = DistilBertForTokenClassification.from_pretrained("distilbert-base-uncased")
    sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
    return tokenizer, ner_model, sbert_model

tokenizer, ner_model, sbert_model = load_models()

# -------------------------------
# Sidebar controls
# -------------------------------
st.sidebar.header("🔍 Analysis Options")

company_index = st.sidebar.selectbox(
    "Select a company record",
    df.index,
    format_func=lambda x: df.loc[x, "Company"]
)

# -------------------------------
# Display structured data
# -------------------------------
st.subheader("🏢 Structured Corporate Information")

structured_cols = [
    "Company",
    "Business_Type",
    "Nature_of_Business",
    "Industry",
    "Founding_Year",
    "Firm_Age",
    "Ownership_Type",
    "Address",
    "Postcode",
    "City",
    "State"
]

st.table(df.loc[[company_index], structured_cols])

# -------------------------------
# Display unstructured text
# -------------------------------
st.subheader("📝 Unstructured Corporate Text")

st.markdown("**Business Description**")
st.write(df.loc[company_index, "Business_Description"])

st.markdown("**Ownership Declaration**")
st.write(df.loc[company_index, "Ownership_Declaration"])

st.markdown("**Narrative Statement**")
st.write(df.loc[company_index, "Narrative_Statement"])

# -------------------------------
# Named Entity Recognition (DistilBERT)
# -------------------------------
st.subheader("🏷️ Extracted Demographic Entities (LLM Tagging)")

text = df.loc[company_index, "Combined_Text"]

inputs = tokenizer(
    text,
    return_tensors="pt",
    truncation=True,
    padding=True,
    max_length=128
)

with torch.no_grad():
    outputs = ner_model(**inputs)

predictions = torch.argmax(outputs.logits, dim=2)[0].numpy()
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

# Map token predictions to label names
# Using DistilBERT default labels: 0 to num_labels-1
# For simplicity, we just label unknown as "LABEL_{id}"
num_labels = outputs.logits.shape[-1]
id2label = {i: f"LABEL_{i}" for i in range(num_labels)}

entities = [
    (token, id2label[pred])
    for token, pred in zip(tokens, predictions)
    if token not in ["[CLS]", "[SEP]", "[PAD]"]
]

st.write("**Token-level predictions (labels):**")
st.write(entities)

# -------------------------------
# Semantic Retrieval (SBERT) - FIXED
# -------------------------------
st.subheader("🔗 Semantic Similarity Analysis (SBERT)")

# Cache embeddings to avoid recomputation
@st.cache_resource
def get_embeddings(corpus, _model):
    # Optional: clean text
    def clean_text(t):
        t = str(t).lower()
        t = re.sub(r'\s+', ' ', t)
        t = re.sub(r'[^\w\s.,]', '', t)
        return t.strip()
    corpus_clean = [clean_text(t) for t in corpus]
    return _model.encode(corpus_clean, convert_to_numpy=True, show_progress_bar=False)

embeddings = get_embeddings(df["Combined_Text"].tolist(), sbert_model)

# Top-K retrieval function
def retrieve_top_k(query_idx, embeddings, df, k=5):
    query_emb = embeddings[query_idx]
    sims = cosine_similarity([query_emb], embeddings)[0]
    sims[query_idx] = -1  # exclude the query itself
    top_k_idx = sims.argsort()[-k:][::-1] #get top-k indices

    #ensure unique company names 
    seen = set()
    unique_indices = []
    for idx in top_k_idx:
        company = df.loc[idx, "Company"]
        if company not in seen:
            seen.add(company)
            unique_indices.append(idx)
        if len(unique_indices) == k:
            break

    return df.iloc[unique_indices], sims[unique_indices]

top_k = st.slider("Number of similar companies to display", 3, 10, 5)

similar_df, sim_scores = retrieve_top_k(company_index, embeddings, df, k=top_k)
similar_df = similar_df[["Company", "Industry", "Ownership_Type"]].copy()
similar_df["Similarity Score"] = sim_scores

st.write("### 🔍 Most Similar Companies")
st.dataframe(similar_df.reset_index(drop=True))

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("Developed for Chapter 3: Modelling, Evaluation and Deployment")