<<<<<<< HEAD
# YouTube Comment Sentiment & Engagement Analyzer

This project is an end-to-end sentiment analysis and content moderation system built using YouTube comments. As an individual Master's student in Data Science, I designed and developed this solution to explore real-world NLP use cases, model interpretability, and scalable deployment using Docker and Streamlit.

---

## 🎯 Project Objective

To build a robust, interpretable sentiment classification system for YouTube comments using advanced Natural Language Processing (NLP) and Machine Learning (ML) techniques, combined with moderation and explainability features.

---

## 🧠 Features & Capabilities

- ✅ Fine-tuned BERT model (`bert-base-uncased`) with backtranslation data
- ✅ PCA for dimensionality reduction
- ✅ Sentiment classification: **Positive**, **Negative**, **Neutral**
- ✅ Rule-based override for moderation (bad word detection)
- ✅ SHAP, LIME, and Captum (Integrated Gradients) explainability
- ✅ Confidence scores, threshold tuning, ROC & PR curves
- ✅ Single comment or batch `.csv` upload prediction
- ✅ Dockerized and deployed via Streamlit UI

---

## 📁 Project Structure

YouTubeSentimentProject/ │ ├── app.py # Streamlit UI ├── sentiment_utils.py # Model, prediction, explainability ├── Dockerfile # Docker build config ├── requirements.txt # Python dependencies ├── README.md # You're reading it now │ ├── model/ # Fine-tuned BERT model ├── data/ # Input/cleaned/test CSVs ├── notebook/ # Jupyter notebook experiments ├── assets/ # Custom visuals (optional) ├── logs/ # Logged moderation cases ├── captum_env/ # Captum integrated gradients ├── test/ # Test case



---

## 📊 Model & Evaluation

- **Embedding**: `all-mpnet-base-v2` from `SentenceTransformers`
- **Class Balancing**: SMOTE & Backtranslation (English → French → English)
- **Classifiers Tried**:
  - Logistic Regression
  - XGBoost
  - BERT + PCA + XGBoost
  - **Final**: Fine-tuned BERT on backtranslated data
- **Evaluation**:
  - Confusion Matrix
  - ROC Curve per class
  - Precision-Recall Curve per class
  - Threshold tuning and confidence-based filtering

---

## 🛡️ Moderation Layer

A rule-based override checks all incoming comments against a 1000+ bad words list (exact + regex matches). If a match is found, the comment is **force-classified as Negative** and logged for audit.

---

## 🧠 Explainability

To ensure model transparency and interpretability, the app integrates:

- **SHAP**: Global feature importance for BERT embeddings
- **LIME**: Local explanations with color-coded word highlights
- **Captum (Integrated Gradients)**: Token-level attribution in HTML format

---

## 🐳 Deployment

This project is fully containerized using Docker:

```bash
docker build -t youtube-sentiment-app .
docker run -p 8501:8501 youtube-sentiment-app


📄 Research Contribution
This project is also documented as a research paper titled:
“Building an Interpretable and Scalable BERT-Based Sentiment Analysis System for YouTube Comments with Explainability and Moderation Features”

It includes:

Model comparisons

Dataset statistics

Screenshot illustrations

Evaluation metrics

Explainability snapshots

Deployment pipeline

🧑‍💻 About Me
I’m currently pursuing my Master's in Data Science and working independently on real-world projects with the aim of joining a leading tech company as a Data Scientist. This project was built to demonstrate my:

Problem-solving skills

Research-backed model experimentation

Proficiency in NLP and ML pipelines

Real-time app deployment experience

🤝 Contact
Feel free to connect or collaborate:

Email: [your email here]

LinkedIn: [your LinkedIn URL]

Research Paper: [add Overleaf link if public]

=======
# youtube-sentiment-analyzer
>>>>>>> 8535f76a351cdca150c4d666e5848deb206a28b7
