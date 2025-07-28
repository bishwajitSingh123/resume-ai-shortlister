# app/main.py
import streamlit as st
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
from utils import (
    extract_text_from_pdf,
    load_sentence_model,
    load_classifier,
    predict_category,
    calculate_match_percentage,
    extract_keywords,
    highlight_keywords_in_resume, 
    generate_ats_score
)

# Hide Streamlit footer & menu
st.markdown("""
    <style>
    #MainMenu, footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Animated circuit background + premium dark theme
st.markdown("""
    <style>
    @keyframes float {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }

    .stApp {
        background: linear-gradient(270deg, #0f0f0f, #1e1e2f, #0f0f0f);
        background-size: 400% 400%;
        animation: float 30s ease infinite;
        color: #e0e0e0;
        font-family: 'Segoe UI', sans-serif;
    }

    h1, h2, h3 {
        color: #00e0ff;
        text-shadow: 0 0 10px #00e0ff;
        font-weight: bold;
    }

    .main {
        background-color: rgba(0, 224, 255, 0.05);
        padding: 3rem;
        border-radius: 20px;
        box-shadow: 0 0 30px rgba(0, 224, 255, 0.3);
        backdrop-filter: blur(10px);
    }

    .block-container {
        padding-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Content
st.markdown("<div class='main'>", unsafe_allow_html=True)
st.title("🕉️ Jai Shree Ram")


# Load models
s_model = load_sentence_model("models/sentence_model")
clf_model, encoder = load_classifier("models/classifier_model.pkl", "models/label_encoder.pkl")

st.set_page_config(page_title="AI Resume Matcher", layout="wide")
st.header("🧠 AI-Powered Resume Shortlisting System")
st.markdown("Upload resumes and a job description to find the best matches 🔍")

# Select JD Input Type
jd_input_type = st.radio("📝 Choose Job Description Input Method:", ["Upload PDF", "Paste Text"])
jd_text = None

if jd_input_type == "Upload PDF":
    jd_file = st.file_uploader("📄 Upload Job Description (PDF)", type=["pdf"], key="jd_file")
    if jd_file:
        jd_text = extract_text_from_pdf(jd_file)
        if not jd_text.strip():
            st.warning("⚠️ JD file seems empty or unreadable.")

elif jd_input_type == "Paste Text":
    jd_text = st.text_area("📝 Paste Job Description Text Here", height=300)

# Upload Resumes
resumes = st.file_uploader("📂 Upload Resumes (PDFs)", accept_multiple_files=True, type=["pdf"], key="resume_upload")

# Helper function for confidence
def get_confidence_label(percentage):    
    if percentage >= 70:
       return "🟢 Strong Fit for this JD"
    elif percentage >= 50:
         return "⚠️ Partial Fit – Improve Keywords"
    else:
        return "❌ Resume needs optimization"

if st.button("🔍 Find Best Matches"):
    if jd_text and resumes:
        jd_category = predict_category(jd_text, s_model, clf_model, encoder)
        results = []

        for resume in resumes:
            res_text = extract_text_from_pdf(resume)
            if not res_text.strip():
                continue

            res_category = predict_category(res_text, s_model, clf_model, encoder)
            match = "✅ Match" if res_category == jd_category else "🔴 Not Match"
            match_percent = calculate_match_percentage(jd_text, res_text, s_model)
            confidence = get_confidence_label(match_percent)
            ats_score = generate_ats_score(res_text, jd_text, s_model)

            results.append({
                "Resume Name": resume.name,
                "Resume": resume,
                "Predicted Category": res_category,
                "Match With JD": match,
                "Match %": match_percent,
                "Confidence": confidence,
                "ATS Score": ats_score,
                "Resume Text": res_text
            })

        cleaned_results = [
            {
                "Resume Name": r["Resume Name"],
                "Predicted Category": r["Predicted Category"],
                "Match With JD": r["Match With JD"],
                "Match %": r["Match %"],
                "Confidence": r["Confidence"],
                "ATS Score": r["ATS Score"]
            }
            for r in results
        ]

        if cleaned_results:
            df = pd.DataFrame(cleaned_results).sort_values(by="Match %", ascending=False)
            st.success(f"🎯 JD Predicted Category: **{jd_category}**")

            st.markdown("### 📊 Top Resume Matches:")
            st.dataframe(df.style.map(
                lambda val: 'color: green' if val == "✅ Match" else 'color: red' if val == "🔴 Not Match" else '',
                subset=["Match With JD"]
            ), use_container_width=True)

            if st.checkbox("✅ Show only matched resumes", key="show_matched_checkbox"):
                df = df[df["Match With JD"] == "✅ Match"]
                st.markdown("### 📊 Filtered Matches")

            if len(df) > 1:
                top_n = st.slider("📌 Show Top N Matches", 1, len(df), min(10, len(df)), key="top_n_slider")
                top_n_df = df.head(top_n)
                st.markdown("### 📌 Top N Matches Table")
                st.dataframe(top_n_df, use_container_width=True)

                for i, row in top_n_df.iterrows():
                    if row['Match %'] >= 60:
                        # st.markdown(f"#### 📄 **{row['Resume Name']}**")
                        # st.markdown(f"**Predicted Category:** {row['Predicted Category']}")
                        # st.markdown(f"**Match %:** {row['Match %']}%")
                        # st.markdown(f"**ATS Score:** {row['ATS Score']}%")
                        # st.markdown(f"**Confidence:** {row['Confidence']}")
                        matched_resume = next((r for r in results if r['Resume Name'] == row['Resume Name']), None)
                        if matched_resume:
                            resume_text = matched_resume['Resume Text']
                            jd_keywords = extract_keywords(jd_text)
                            highlighted = highlight_keywords_in_resume(resume_text, jd_keywords)
                            st.markdown("##### 🔍 Resume View:", unsafe_allow_html=True)
                            st.markdown(highlighted, unsafe_allow_html=True)

            else:
                st.info("🔍 Only 1 matching resume found.")        

            # st.markdown("## 📝 Resume Keyword Highlights")

            # for resume, row in zip(resumes, cleaned_results):
            #     if row.get("Match With JD") == "✅ Match":
            #         resume_bytes = resume.read()
            #         resume.seek(0)
            #         resume_text = extract_text_from_pdf(resume)
            #         jd_keywords = extract_keywords(jd_text)
            #         highlighted = highlight_keywords_in_resume(resume_text, jd_keywords)

            #         st.markdown(f"### 📄 {resume.name}", unsafe_allow_html=True)
            #         st.markdown(highlighted, unsafe_allow_html=True)

            #         st.download_button(
            #             label=f"⬇️ Download Resume - {resume.name}",
            #             data=resume_bytes,
            #             file_name=resume.name,
            #             mime="application/pdf"
            #         )

            df.to_csv("output/top_resume_matches.csv", index=False)
            st.download_button("📥 Download Result CSV", df.to_csv(index=False), "top_resume_matches.csv", "text/csv")

            if row["Match With JD"] == "✅ Match":
                st.markdown("### 📂 Download Matching Resumes")
                for resume, row in zip(resumes, results):
                    resume_bytes = resume.read()
                    st.download_button(
                        label=f"⬇️ Download {resume.name}",
                        data=resume_bytes,
                        file_name=resume.name,
                        mime="application/pdf"
                    )
            else:
                st.markdown("No Resume Matched Jd exactly.")

            # st.markdown("### 📈 Match Summary")
            # match_count = df["Match With JD"].value_counts()
            # fig, ax = plt.subplots(figsize=(0.5, 0.5))  # Reduced size to ~60%
            # colors = ['#8BC34A', '#F44336']
            # match_count.plot.pie(
            #     autopct='%1.1f%%',
            #     colors=colors,
            #     startangle=90,
            #     textprops={'fontsize': 10},
            #     ax=ax,
            #     wedgeprops={'edgecolor': 'white'}
            # )
            # ax.set_ylabel('')
            # ax.set_title("Resume Match Distribution", fontsize=12)
            # st.pyplot(fig)

            # st.subheader("📊 Match Summary (Top N)")
            fig, ax = plt.subplots(figsize=(2.5, 2.5))  # Reduced 50% size from typical 5x5
            labels = ['Strong Fit', 'Moderate Fit', 'Low Fit']
            sizes = [50, 30, 20]
            colors = ['#4CAF50', '#FFC107', '#F44336']
            ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            st.pyplot(fig)


        else:
            st.warning("⚠️ No valid resume text could be extracted. Please upload readable PDFs.")
        from utils import store_feedback  # make sure this is at the top of main.py

        st.markdown("### 🧠 Feedback on Prediction")
        with st.form("feedback_form"):
            st.markdown("Help us improve by giving feedback on the prediction:")

            feedback_actual_label = st.text_input("Correct Category (if different):")
            feedback_decision = st.selectbox("Was the predicted category correct?", ["Yes", "No"])
            feedback_comment = st.text_area("Any additional feedback (optional):", height=100)

            submit_feedback = st.form_submit_button("Submit Feedback")

            # Ensure these variables are set BEFORE the feedback form
            # uploaded_jd_text = []  # or text_area/browse result
            # top_resume_text = []
            # predicted_label = []  # or whatever your prediction output is


            if submit_feedback:
                if jd_text and resume_text:
                    actual = feedback_actual_label if feedback_actual_label else "N/A"
                    store_feedback(
                        jd_text=jd_text,
                        resume_text=res_text,
                        actual_label=actual,
                        predicted_label=predict_category
                    )
                    st.success("✅ Feedback submitted successfully. Thank you!")
                else:
                    st.warning("⚠️ Please upload a JD and a resume to give feedback.")

    else:
        st.error("⚠️ Please upload/paste JD and upload at least one resume.")
