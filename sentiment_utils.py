import re
import numpy as np
import pandas as pd
import torch
import os
import requests
import gdown
import streamlit as st
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, BertForSequenceClassification
from sklearn.preprocessing import LabelEncoder
import csv
from datetime import datetime
from scipy.special import softmax

# -----------------------------
# 0. LOAD BAD WORD LISTS
# -----------------------------
def load_bad_words(file_path="data/Final/bad_words.txt"):
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return set(word.strip().lower() for word in f if word.strip())
    return set()

def load_bad_word_patterns(file_path="data/Final/bad_words_regex.txt"):
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return [re.compile(p.strip(), re.IGNORECASE) for p in f if p.strip()]
    return []

BAD_WORDS = load_bad_words()
BAD_WORD_PATTERNS = load_bad_word_patterns()

def contains_bad_words(text):
    words = text.split()
    if any(word in BAD_WORDS for word in words):
        return True
    for pattern in BAD_WORD_PATTERNS:
        if pattern.search(text):
            return True
    return False

# -----------------------------
# 1. CLEAN TEXT
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# -----------------------------
# 2. LOAD BERT MODEL AND TOKENIZER
# -----------------------------
bert_model_path = "model/fine_tuned_bert"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    print("📁 Model folder contents:", os.listdir(bert_model_path))
    bert_model = BertForSequenceClassification.from_pretrained(
        bert_model_path,
        local_files_only=True
    ).to(device)
    bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_path)
except Exception as e:
    import traceback
    print("❌ FULL TRACEBACK:\n", traceback.format_exc())
    raise RuntimeError(f"❌ Failed to load BERT model or tokenizer. Error: {e}")

# ✅ Predict function for a list of texts
def predict_proba_bert(texts, batch_size=32):
    all_probs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = bert_tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=256).to(device)
        with torch.no_grad():
            outputs = bert_model(**inputs)
            probs = softmax(outputs.logits.cpu().numpy(), axis=1)
            all_probs.extend(probs)
    return np.array(all_probs)

# -----------------------------
# 3. SINGLE TEXT PREDICTION
# -----------------------------
def predict_single_text(text, label_encoder, return_proba=False):
    cleaned_text = clean_text(text)

    if contains_bad_words(cleaned_text):
        log_flagged_comment(text, cleaned_text, match_type="bad_word", source="single")
        if return_proba:
            return np.array([1.0, 0.0, 0.0])
        return np.array([1.0, 0.0, 0.0]), "Negative", 1.0

    probs = predict_proba_bert([cleaned_text])[0]
    pred_class = np.argmax(probs)
    label = label_encoder.inverse_transform([pred_class])[0]
    confidence = np.max(probs)

    if return_proba:
        return probs
    return probs, label, confidence

# -----------------------------
# 4. BATCH PREDICTION
# -----------------------------
def predict_batch_csv(uploaded_file, label_encoder):
    df = pd.read_csv(uploaded_file)
    if "comment" not in df.columns:
        raise ValueError("CSV must have a 'comment' column")

    df["cleaned"] = df["comment"].astype(str).apply(clean_text)
    predictions = []
    confidences = []

    for text in df["cleaned"]:
        if contains_bad_words(text):
            predictions.append("Negative")
            confidences.append(1.0)
        else:
            probs = predict_proba_bert([text])[0]
            pred = label_encoder.inverse_transform([np.argmax(probs)])[0]
            predictions.append(pred)
            confidences.append(np.max(probs))

    df["Predicted Label"] = predictions
    df["Confidence Score"] = confidences
    return df, df

# -----------------------------
# 5. LOGGING BAD COMMENTS
# -----------------------------
def log_flagged_comment(original_text, cleaned_text, match_type, source="single"):
    log_path = os.path.join("logs", "flagged_comments_log.csv")
    os.makedirs("logs", exist_ok=True)
    fields = [original_text, cleaned_text, match_type, source, datetime.now().isoformat()]
    headers = ["original_text", "cleaned_text", "match_type", "source", "timestamp"]
    write_header = not os.path.exists(log_path)

    with open(log_path, mode="a", encoding="utf-8", newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(headers)
        writer.writerow(fields)

# -----------------------------
# 6. CAPTUM EXPLANATION
# -----------------------------
from captum.attr import IntegratedGradients, visualization
from IPython.core.display import display, HTML  # Needed for HTML rendering outside notebooks

def explain_with_captum(text, model, tokenizer, device):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    def get_embeddings(input_ids):
        return model.bert.embeddings.word_embeddings(input_ids)

    input_embed = get_embeddings(input_ids)
    baseline_embed = torch.zeros_like(input_embed)

    def forward_func(embeddings):
        output = model(inputs_embeds=embeddings, attention_mask=attention_mask)
        return torch.softmax(output.logits, dim=1)

    with torch.no_grad():
        probs = forward_func(input_embed)
        pred_class = torch.argmax(probs).item()

    ig = IntegratedGradients(forward_func)
    attributions, delta = ig.attribute(
        inputs=input_embed,
        baselines=baseline_embed,
        target=pred_class,
        return_convergence_delta=True
    )

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    tokens = [t for t in tokens if t not in ("[CLS]", "[SEP]")]

    attributions_sum = attributions.sum(dim=-1).squeeze(0)
    attributions_sum = attributions_sum[:len(tokens)]
    attributions_sum = attributions_sum / torch.norm(attributions_sum)

    vis_data = visualization.VisualizationDataRecord(
        word_attributions=attributions_sum.detach().cpu().numpy(),
        pred_prob=probs[0][pred_class].item(),
        pred_class=pred_class,
        true_class=pred_class,
        attr_class="Predicted",
        attr_score=attributions_sum.sum().item(),
        raw_input_ids=tokens,
        convergence_score=delta.sum().item()
    )

    # ✅ Return HTML string explicitly
    html = visualization.visualize_text([vis_data])._repr_html_()
    return html
