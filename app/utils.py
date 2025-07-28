# app/utils.py
import os
import re
import fitz  # PyMuPDF
import joblib
import numpy as np
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer


# Load transformer model
def load_sentence_model(path="models/sentence_model"):
    return SentenceTransformer(path)

def extract_text_from_pdf(file_obj):
    doc = fitz.open(stream=file_obj.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def extract_keywords_from_jd(jd_text, top_n=20):
    vectorizer = CountVectorizer(stop_words='english')
    word_counts = vectorizer.fit_transform([jd_text])
    word_freq = dict(zip(vectorizer.get_feature_names_out(), word_counts.toarray().flatten()))
    sorted_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [kw[0] for kw in sorted_keywords[:top_n]]

def extract_text_from_pdf(uploaded_file):
    text = ""
    reader = PdfReader(uploaded_file)
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def load_classifier(path="models/classifier_model.pkl", encoder_path="models/label_encoder.pkl"):
    model = joblib.load(path)
    encoder = joblib.load(encoder_path)
    return model, encoder

def get_embedding(text, model):
    return model.encode([text])[0]

def predict_category(text, s_model, clf_model, encoder):
    vector = get_embedding(text, s_model).reshape(1, -1)
    prediction = clf_model.predict(vector)
    label = encoder.inverse_transform(prediction)
    return label[0]

def calculate_match_percentage(jd_text, resume_text, s_model):
    jd_vec = get_embedding(jd_text, s_model).reshape(1, -1)
    res_vec = get_embedding(resume_text, s_model).reshape(1, -1)
    similarity = cosine_similarity(jd_vec, res_vec)[0][0]
    return round(similarity * 100, 2)  # returns percentage

import re

def extract_keywords(text, top_k=10):
    words = re.findall(r'\b\w+\b', text.lower())
    stopwords = set(['the', 'and', 'to', 'for', 'in', 'of', 'on', 'a', 'an', 'with', 'by','am', 'is', 'are', 'was', 'were', 'have', 'has', 'do', 'does', 'did', 'shall', 'should', 'can', 'could','will', 'would' 'must', 'out', 'need','used', 'to'])  # minimal set
    filtered = [word for word in words if word not in stopwords and len(word) > 2]
    freq = {}
    for word in filtered:
        freq[word] = freq.get(word, 0) + 1
    sorted_keywords = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [kw for kw, count in sorted_keywords[:top_k]]

def highlight_keywords_in_resume(resume_text, jd_text):
    jd_combined = " ".join(jd_text) if isinstance(jd_text, list) else jd_text
    keywords = extract_keywords(jd_combined)
    highlights = {kw: resume_text.lower().count(kw) for kw in keywords}
    return highlights  # returns dict of keyword: count in resume

def generate_ats_score(resume_text, jd_text, s_model):
    keyword_hits = highlight_keywords_in_resume(resume_text, jd_text)
    keyword_score = sum(1 for count in keyword_hits.values() if count > 0) / len(keyword_hits)
    
    match_percent = calculate_match_percentage(jd_text, resume_text, s_model) / 100
    ats_score = round((0.6 * match_percent + 0.4 * keyword_score) * 100, 2)
    return ats_score

def store_feedback(jd_text, resume_text, actual_label, predicted_label, feedback_path="feedback.csv"):
    import pandas as pd
    row = {
        "jd_text": jd_text,
        "resume_text": resume_text,
        "actual_label": actual_label,
        "predicted_label": predicted_label
    }
    if os.path.exists(feedback_path):
        df = pd.read_csv(feedback_path)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(feedback_path, index=False)
