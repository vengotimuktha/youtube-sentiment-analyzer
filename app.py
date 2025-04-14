import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
from lime.lime_text import LimeTextExplainer
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertForSequenceClassification

from sentiment_utils import (
    clean_text, predict_single_text, predict_batch_csv, contains_bad_words,
    explain_with_captum, predict_proba_bert
)

# ---------------------- INITIAL SETUP ----------------------
label_encoder = LabelEncoder().fit(["Negative", "Neutral", "Positive"])
class_names = label_encoder.classes_.tolist()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = "model/fine_tuned_bert"
model = BertForSequenceClassification.from_pretrained(model_path).to(device)
tokenizer = BertTokenizer.from_pretrained(model_path)
model.eval()

lime_explainer = LimeTextExplainer(class_names=class_names)

if "threshold" not in st.session_state:
    st.session_state.threshold = 0.5

# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(page_title="YouTube Sentiment Analyzer (BERT)", layout="wide")

# --- Logo Image Header ---
st.image("assets/logo.png", width=150)  

st.markdown("<h1 style='margin-top: -10px;'>YouTube Comment Sentiment Analyzer</h1>", unsafe_allow_html=True)

st.markdown("""
Fine-tuned BERT sentiment classifier with confidence thresholding, **LIME** and **Captum (IG)** explainability, and bad-word moderation logging.
""")


# ---------------------- NAVIGATION ----------------------
st.sidebar.title(" Navigation: ")
section = st.sidebar.radio("Go to:", [
    "Single Comment Prediction", 
    "Batch Upload (.csv)",
    "Visual Analytics",
    "View Moderation Logs",
    "Settings"
])

# ---------------------- SINGLE COMMENT PREDICTION ----------------------
if section == "Single Comment Prediction":
    st.subheader("Predict Sentiment")

    keys = [
        "pred_label", "pred_confidence", "pred_class",
        "cleaned", "html_plot", "pred_probs"
    ]
    for k in keys:
        if k not in st.session_state:
            st.session_state[k] = None

    def clear_all():
        for k in keys + ["comment_input"]:
            st.session_state[k] = None
        st.rerun()

    with st.form("comment_form"):
        user_input = st.text_area(
            "Enter YouTube comment:",
            key="comment_input",
            height=100
        )
        submitted = st.form_submit_button(" Predict Sentiment")

        if submitted and user_input:
            cleaned = clean_text(user_input)
            probs, label, confidence = predict_single_text(cleaned, label_encoder)
            pred_class = np.argmax(probs)

            st.session_state.pred_probs = probs
            st.session_state.pred_label = label
            st.session_state.pred_confidence = confidence
            st.session_state.pred_class = pred_class
            st.session_state.cleaned = cleaned
            st.session_state.html_plot = None

    st.button("Clear Text", on_click=clear_all)

    if st.session_state.pred_label is not None:
        label = st.session_state.pred_label
        confidence = st.session_state.pred_confidence

        if confidence == 1.0 and label == "Negative" and contains_bad_words(st.session_state.cleaned):
            st.warning("⚠️ Offensive content detected. Flagged as Negative.")

        if confidence >= st.session_state.threshold:
            st.success(f"**Predicted Sentiment**: {label}")
        else:
            st.warning("Prediction: **Uncertain** (low confidence)")

        st.info(f"Confidence Score: **{confidence:.2f}**")

        st.subheader("LIME Explanation")
        def lime_fn(texts):
            return predict_proba_bert([clean_text(t) for t in texts])
        exp = lime_explainer.explain_instance(
            st.session_state.comment_input, lime_fn, num_features=10, labels=[0, 1, 2]
        )
        st.components.v1.html(exp.as_html(labels=(st.session_state.pred_class,)), height=520, scrolling=True)

        st.subheader("Captum (Integrated Gradients) Explanation")
        with st.expander("Show Captum Explanation"):
            if st.session_state.html_plot is None:
                html_plot = explain_with_captum(
                    st.session_state.comment_input, model, tokenizer, device
                )
                st.session_state.html_plot = html_plot

            st.components.v1.html(st.session_state.html_plot, height=400, scrolling=True)
            st.download_button(
                label="📅 Download Captum Report",
                data=st.session_state.html_plot,
                file_name="captum_explanation.html",
                mime="text/html"
            )

# ---------------------- BATCH PREDICTION ----------------------
elif section == "Batch Upload (.csv)":
    st.subheader("📂 Upload CSV")
    uploaded_file = st.file_uploader("Upload a CSV with a 'comment' column", type=["csv"])

    if uploaded_file:
        try:
            df, results_df = predict_batch_csv(uploaded_file, label_encoder)
        except Exception as e:
            st.error(f"Error: {e}")
        else:
            st.write("Prediction Results")
            output_df = results_df[["comment", "Predicted Label", "Confidence Score"]]
            st.dataframe(output_df)
            csv_data = output_df.to_csv(index=False).encode("utf-8")
            st.download_button("📅 Download Results", csv_data, "predictions.csv", "text/csv")

# ---------------------- VISUAL ANALYTICS ----------------------
elif section == "Visual Analytics":
    st.subheader("Confidence Analysis")
    uploaded_csv = st.file_uploader("Upload a CSV of predictions", type=["csv"], key="viz")
    default_path = os.path.join("data", "Final", "confidence_analysis.csv")

    if uploaded_csv:
        df = pd.read_csv(uploaded_csv)
    elif os.path.exists(default_path):
        df = pd.read_csv(default_path)
        st.caption("Using default sample results.")
    else:
        st.error("No data available.")
        st.stop()

    if "Predicted Label" not in df.columns or "Confidence Score" not in df.columns:
        st.error("File must include 'Predicted Label' and 'Confidence Score' columns.")
        st.stop()

    st.bar_chart(df["Predicted Label"].value_counts())
    st.subheader("📉 Confidence Distribution")
    fig, ax = plt.subplots()
    ax.hist(df["Confidence Score"], bins=20, color="skyblue", edgecolor="black")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

# ---------------------- VIEW LOGS ----------------------
elif section == "View Moderation Logs":
    st.subheader("Flagged Comments")
    log_path = os.path.join("logs", "flagged_comments_log.csv")

    if os.path.exists(log_path):
        logs_df = pd.read_csv(log_path)
        st.dataframe(logs_df)

        st.download_button("📅 Download Logs", logs_df.to_csv(index=False).encode("utf-8"), "flagged_logs.csv")

        if st.button("🩹 Clear All Logs"):
            with open(log_path, "w", encoding="utf-8") as f:
                f.write("original_text,cleaned_text,match_type,source,timestamp\n")
            st.success("✅ All logs cleared successfully! Please refresh to see changes.")
    else:
        st.info("! No logs found.")

# ---------------------- SETTINGS ----------------------
elif section == "⚙️ Settings":
    st.subheader("⚙️ Adjust Model Threshold")
    new_threshold = st.slider("Classification Confidence Threshold", 0.30, 0.95, st.session_state.threshold)
    st.session_state.threshold = new_threshold
    st.info(f"Model will show results as **Uncertain** if confidence < {new_threshold:.2f}")

# ---------------------- FOOTER ----------------------
st.markdown("""---  
Built by Mukthasree Vengoti  
[🔗 GitHub](https://github.com/vengotimuktha/youtube-sentiment-analyzer)
""")