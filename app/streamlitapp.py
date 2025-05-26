import streamlit as st
import os
import numpy as np
import tensorflow as tf
from tempfile import NamedTemporaryFile
from utils import load_video, num_to_char
from modelutil import load_model

# ------------------ Page Configuration ------------------
st.set_page_config(
    page_title="LipLens | AI-Powered Lip Reading",
    layout="wide",
    page_icon="💬"
)


# ------------------ Professional Styling ------------------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    [data-testid="stMainBlockContainer"] {
        font-family: 'Inter', sans-serif;
        background-image: url("https://www.shutterstock.com/image-vector/abstract-ai-circuit-board-background-600nw-2471339475.jpg");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        backdrop-filter: blur(8px);
        padding: 2rem;
    }

    .title {
        font-size: 3rem;
        font-weight: 700;
        color: #ffffff;
        text-shadow: 0 0 15px rgba(0, 128, 255, 0.8);
        margin-bottom: 0.5rem;
        letter-spacing: 1px;
        animation: fadeInDown 1s ease-out;
        text-align: center;
    }

    .subtitle {
        font-size: 1.25rem;
        font-weight: 400;
        color: #e2e8f0;
        margin-bottom: 2.5rem;
        animation: fadeInUp 1.3s ease-out;
        text-align: center;
    }

    .upload-box {
        border: 2px solid #3b82f6;
        border-radius: 12px;
        padding: 1.5rem;
        background-color: rgba(255, 255, 255, 0.9);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
        transition: transform 0.3s, box-shadow 0.3s;
        margin: 2rem 0;
    }

    .upload-box:hover {
        transform: scale(1.02);
        box-shadow: 0 6px 30px rgba(0, 0, 0, 0.2);
    }

    .section-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: white;
        margin-bottom: 1rem;
        text-transform: uppercase;
        text-align: center;
    }

    .pred-box {
        background-color: rgba(241, 245, 249, 0.95);
        padding: 1.5rem;
        border-left: 5px solid #3b82f6;
        border-radius: 10px;
        font-size: 1.1rem;
        font-weight: 500;
        color: #0f172a;
        margin-top: 1.5rem;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    }

    .footer {
        margin-top: 4rem;
        text-align: center;
        font-size: 0.9rem;
        color: #cbd5e1;
        opacity: 0.9;
        padding: 1rem 0;
    }
    
    .centered-container {
        max-width: 800px;
        margin: 0 auto;
        padding: 2rem;
    }

    @keyframes fadeInDown {
        0% { opacity: 0; transform: translateY(-20px); }
        100% { opacity: 1; transform: translateY(0); }
    }

    @keyframes fadeInUp {
        0% { opacity: 0; transform: translateY(20px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    </style>
""", unsafe_allow_html=True)


# ------------------ Header ------------------
st.markdown('<div class="title">LipLens</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-Powered Speech Recognition from Lip Movements</div>', unsafe_allow_html=True)

# ------------------ Upload Section ------------------
st.markdown('<div class="upload-box">', unsafe_allow_html=True)
uploaded_file = st.file_uploader("📁 Upload a `.mpg` video file", type=["mpg"])
st.markdown('</div>', unsafe_allow_html=True)

# ------------------ Prediction Workflow ------------------
if uploaded_file is not None:
    st.success("Upload successful.")

    with NamedTemporaryFile(delete=False, suffix=".mpg") as temp_file:
        temp_file.write(uploaded_file.read())
        file_path = temp_file.name

    st.markdown('<div class="centered-container">', unsafe_allow_html=True)

    st.markdown("### Preview & Analysis")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<div class="section-title">🎬 Video Preview</div>', unsafe_allow_html=True)
        os.system(f'ffmpeg -i "{file_path}" -vcodec libx264 temp_video.mp4 -y')
        with open("temp_video.mp4", "rb") as video_file:
            st.video(video_file.read())

    with col2:
        st.markdown('<div class="section-title">🤖 Model Prediction</div>', unsafe_allow_html=True)
        with st.spinner("Processing video..."):
            video_tensor = load_video(file_path)
            video_tensor = tf.cast(video_tensor, tf.float32)

            model = load_model()
            yhat = model.predict(tf.expand_dims(video_tensor, axis=0))
            decoded = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
            predicted_text = tf.strings.reduce_join(num_to_char(decoded)).numpy().decode("utf-8")

        st.markdown(f"<div class='pred-box'>📝 <strong>Transcript:</strong><br>{predicted_text}</div>", unsafe_allow_html=True)

else:
    st.info("Please upload a `.mpg` video to get started.")

# ------------------ Footer ------------------
st.markdown('<div class="footer">© 2025 LipLens AI — Built with TensorFlow & Streamlit</div>', unsafe_allow_html=True)

