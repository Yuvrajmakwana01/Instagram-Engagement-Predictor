import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load model and scaler
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Set page
st.set_page_config(page_title="📲 Instagram Engagement Predictor", layout="centered")
st.title("📲 Instagram Engagement Predictor")
st.markdown("Enter your Instagram post details to estimate the expected number of likes.")

# Inputs
followers = st.number_input("👥 Number of Followers", min_value=0, step=1)
caption = st.text_area("📝 Caption")
hashtags = st.text_area("🏷️ Hashtags (comma-separated)")
posted_hours = st.number_input("🕒 Hours Since Posted", min_value=0.0, step=1.0)

# Feature engineering
caption_length = len(caption.strip())
hashtag_count = len([tag.strip() for tag in hashtags.split(',') if tag.strip() != ""])

# Predict
if st.button("📊 Predict Likes"):
    if followers == 0:
        st.warning("Please enter at least some followers.")
    else:
        input_df = pd.DataFrame([{
            "Followers": followers,
            "CaptionLength": caption_length,
            "HashtagCount": hashtag_count,
            "PostedHoursAgo": posted_hours
        }])

        scaled_input = scaler.transform(input_df)
        predicted_likes = model.predict(scaled_input)[0]
        st.success(f"📈 Estimated Likes: {int(round(predicted_likes))}")
