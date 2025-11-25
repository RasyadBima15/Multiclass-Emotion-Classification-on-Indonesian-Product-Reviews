import streamlit as st
import pandas as pd
import numpy as np
import re
from collections import Counter
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import time
import base64
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import numpy as np

model_path = "Rasyy/indobert_multiclass_emotion_classifier_for_indonesian_e_commerce_reviews"  # Pastikan path ini benar
MAX_LENGTH = 256
LABEL_MAP = {
    0: 'Anger',
    1: 'Fear',
    2: 'Happy',
    3: 'Love',
    4: 'Sad'
}

# --- Fungsi Memuat Model dan Tokenizer ---
# Gunakan st.cache_resource agar model hanya dimuat sekali
@st.cache_resource
def load_model_and_tokenizer(path):
    """Memuat tokenizer dan model IndoBERT dari folder yang ditentukan."""
    try:
        # Memuat Tokenizer
        print("Memuat tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(path)

        # Memuat Model TFBertForSequenceClassification
        print("Memuat model...")
        # from_pt=True digunakan jika model di folder disimpan sebagai PyTorch
        model = AutoModelForSequenceClassification.from_pretrained(
            path
        )

        print("Model dan Tokenizer siap digunakan!")
        return tokenizer, model
    except Exception as e:
        st.error(f"Gagal memuat model atau tokenizer dari {path}. Pastikan folder ada dan berisi file yang diperlukan. Error: {e}")
        return None, None

# --- Memuat Model Global ---
tokenizer, model = load_model_and_tokenizer(model_path)

# --- Fungsi Prediksi ---
def predict_emotion(text, tokenizer, model, max_length=MAX_LENGTH, label_map=LABEL_MAP):
    """Melakukan prediksi emosi pada satu teks menggunakan PyTorch model."""
    if not model or not tokenizer:
        return "Model belum dimuat.", None

    # 1. Tokenisasi Input
    inputs = tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        padding='max_length',
        return_tensors='pt'  # PyTorch tensor
    )

    # 2. Prediksi Model
    with torch.no_grad():  # matikan gradient, lebih cepat
        outputs = model(**inputs)
        logits = outputs.logits  # shape: [1, num_classes]

    # 3. Hitung Probabilitas dan Prediksi Kelas
    probabilities = F.softmax(logits, dim=-1).squeeze().cpu().numpy()  # numpy array
    predicted_class_id = torch.argmax(logits, dim=1).item()  # integer
    predicted_label = label_map.get(predicted_class_id, "Unknown")

    # Format probabilitas untuk ditampilkan per label
    results = {label: float(prob) for label, prob in zip(label_map.values(), probabilities)}

    return predicted_label, results

# --- Konfigurasi Aplikasi Streamlit ---
st.set_page_config(
    layout="wide",
    page_title="Multiclass Emotion Classifier for Indonesian E-Commerce Reviews"
)

def home_page():
    """Halaman Profil Singkat Aplikasi."""
    st.title("🏠 Home")
    st.subheader("🧠 Multiclass Emotion Classifier for Indonesian E-Commerce Reviews")

    st.markdown("""
    ### 🎯 **Tujuan Aplikasi**
    Aplikasi ini dikembangkan sebagai bagian dari penelitian berjudul **"Analisis Performa Pendekatan Zero-Shot dan Few-Shot Learning pada Model Generative AI vs Fine-Tuning BERT dalam Klasifikasi Multiclass Emosi Ulasan Produk E-Commerce Indonesia."**
    
    Aplikasi ini digunakan untuk:
    - Menguji dan membandingkan performa berbagai model **Generative AI** dan **BERT Fine-Tuning**
    - Mengukur metrik **efisiensi (Precision, Recall, F1-Score, Accuracy)** serta **efektivitas (Latency & Throughput)**
    - Mengimplementasikan prediksi emosi ulasan produk berbahasa Indonesia melalui dua skenario penggunaan: **teks tunggal** dan **teks massal**
    """)

    st.markdown("---")

    st.markdown("""
    ### 📦 **Dataset yang Digunakan**
    Aplikasi ini menggunakan dataset **PRDECT-ID (Indonesian Product Reviews Dataset for Emotion Classification Tasks)**.

    Dataset ini berisi ulasan produk e-commerce berbahasa Indonesia dengan 5 label emosi:
    **Anger (Marah), Fear (Takut), Sadness (Sedih), Happy (Senang), dan Love (Suka).**

    Dataset PRDECT-ID dikumpulkan secara langsung melalui website Tokopedia yang berisi ulasan produk dari 29 kategori produk pada platform Tokopedia yang ditulis dalam bahasa Indonesia.

    Dataset dapat diakses melalui publikasi ilmiah:  
    *https://doi.org/10.1016/j.dib.2022.108554*
    """)

    st.markdown("---")

    st.markdown("""
    ### 🚀 **Fitur Aplikasi**
    | Fitur | Deskripsi |
    |-------|-----------|
    | **📊 Hasil Analisis Model** | Menampilkan perbandingan performa model berdasarkan metrik efisiensi & efektivitas |
    | **✍️ Prediksi Teks Tunggal** | Prediksi emosi untuk satu ulasan secara langsung |
    | **📁 Prediksi Teks Massal** | Prediksi banyak ulasan sekaligus menggunakan file CSV |

    ### 🧪 **Model yang Dibandingkan**
    **Generative AI**
    - GPT-5.1  
    - Gemini 2.5 Flash  
    - Gemma-3 (12B)  
    - LLaMA-3.1 (8B)

    **Fine-Tuning BERT-Based Models**
    - IndoBERT
    - RoBERTa
    - DistilBERT
    - mBERT
    """)     

def model_analysis_page():
    """Halaman Analisis Performa Model."""
    st.title("📊 Hasil Analsis Model")
    st.markdown("""
    Halaman ini menampilkan **perbandingan performa model** berdasarkan:
    - Efektivitas: Precision, Recall, F1-Score, Accuracy  
    - Efisiensi: Latency & Throughput  
    - Perbandingan waktu pelatihan (Fine-tuning BERT)  
    - Loss & F1-Score per Epoch (Fine-tuning BERT)
    """)

    st.divider()

    # =======================
    # MODEL PERFORMANCE TABLE
    # =======================

    def highlight_max(s):
        is_max = s == s.max()
        return [
            'background-color: #FFE97F; color: #000000;' if v else ''
            for v in is_max
        ]

    st.subheader("📈 Perbandingan Presisi per Label")
    precision_data = {
        "Model": [
            "DistilBERT", "RoBERTa", "mBERT", "IndoBERT",
            "Gemini 2.5 Flash (0-shot)", "Gemini 2.5 Flash (5-shot)", "Gemini 2.5 Flash (10-shot)",
            "Gemini 2.5 Flash (15-shot)", "Gemini 2.5 Flash (20-shot)", "Gemini 2.5 Flash (25-shot)",
            "GPT-5.1 (0-shot)", "GPT-5.1 (5-shot)", "GPT-5.1 (10-shot)", "GPT-5.1 (15-shot)",
            "GPT-5.1 (20-shot)", "GPT-5.1 (25-shot)",
            "Llama3.1-8B (0-shot)", "Llama3.1-8B (5-shot)", "Llama3.1-8B (10-shot)", "Llama3.1-8B (15-shot)",
            "Llama3.1-8B (20-shot)", "Llama3.1-8B (25-shot)",
            "Gemma3-12B (0-shot)", "Gemma3-12B (5-shot)", "Gemma3-12B (10-shot)", "Gemma3-12B (15-shot)",
            "Gemma3-12B (20-shot)", "Gemma3-12B (25-shot)"
        ],

        # LABEL SCORES
        "Anger":   [0.69, 0.75, 0.71, 0.80, 0.54, 0.60, 0.57, 0.63, 0.62, 0.72, 0.60, 0.59, 0.62, 0.60, 0.64, 0.70, 0.56, 0.50, 0.56, 0.59, 0.53, 0.62, 0.53, 0.69, 0.67, 0.64, 0.64, 0.66],
        "Fear":    [0.56, 0.54, 0.57, 0.60, 0.00, 0.40, 0.67, 0.67, 0.50, 0.67, 1.00, 0.50, 0.49, 0.45, 0.48, 0.49, 0.00, 0.30, 0.36, 0.43, 0.40, 0.31, 0.00, 0.50, 0.43, 0.50, 0.33, 0.33],
        "Happy":   [0.83, 0.76, 0.79, 0.79, 0.71, 0.73, 0.72, 0.77, 0.73, 0.74, 0.60, 0.73, 0.76, 0.77, 0.74, 0.74, 0.51, 0.55, 0.56, 0.57, 0.67, 0.61, 0.62, 0.72, 0.59, 0.62, 0.68, 0.65],
        "Love":    [0.85, 0.84, 0.82, 0.90, 0.93, 0.88, 0.83, 0.85, 0.86, 0.84, 0.89, 0.82, 0.85, 0.80, 0.80, 0.80, 1.00, 0.67, 0.71, 0.75, 0.73, 0.70, 0.86, 0.85, 0.88, 0.77, 0.81, 0.88],
        "Sadness": [0.67, 0.69, 0.66, 0.68, 0.49, 0.47, 0.47, 0.48, 0.48, 0.46, 0.46, 0.50, 0.47, 0.52, 0.52, 0.50, 0.47, 0.51, 0.51, 0.48, 0.49, 0.45, 0.48, 0.47, 0.47, 0.49, 0.44, 0.43],

        # MACRO AVG
        "Precision Macro Avg": [
            0.72, 0.72, 0.71, 0.75,
            0.53, 0.62, 0.65, 0.68, 0.64, 0.68,
            0.71, 0.63, 0.64, 0.63, 0.64, 0.65,
            0.51, 0.51, 0.54, 0.57, 0.56, 0.54,
            0.50, 0.65, 0.61, 0.60, 0.58, 0.59
        ]
    }
    df_precision = pd.DataFrame(precision_data)
    styled_df = df_precision.style.apply(highlight_max, subset=[
        "Anger", "Fear", "Happy", "Love", "Sadness", "Precision Macro Avg"
    ])
    st.dataframe(styled_df, use_container_width=True)
    st.divider()

    st.subheader("📈 Perbandingan Recall per Label")
    recall_data = {
        "Model": [
            "DistilBERT", "RoBERTa", "mBERT", "IndoBERT",
            "Gemini 2.5 Flash (0-shot)", "Gemini 2.5 Flash (5-shot)", "Gemini 2.5 Flash (10-shot)",
            "Gemini 2.5 Flash (15-shot)", "Gemini 2.5 Flash (20-shot)", "Gemini 2.5 Flash (25-shot)",
            "GPT-5.1 (0-shot)", "GPT-5.1 (5-shot)", "GPT-5.1 (10-shot)", "GPT-5.1 (15-shot)",
            "GPT-5.1 (20-shot)", "GPT-5.1 (25-shot)",
            "Llama3.1-8B (0-shot)", "Llama3.1-8B (5-shot)", "Llama3.1-8B (10-shot)", "Llama3.1-8B (15-shot)",
            "Llama3.1-8B (20-shot)", "Llama3.1-8B (25-shot)",
            "Gemma3-12B (0-shot)", "Gemma3-12B (5-shot)", "Gemma3-12B (10-shot)", "Gemma3-12B (15-shot)",
            "Gemma3-12B (20-shot)", "Gemma3-12B (25-shot)"
        ],

        "Anger": [
            0.66, 0.62, 0.59, 0.66,
            0.80, 0.73, 0.69, 0.71, 0.69, 0.62,
            0.60, 0.67, 0.64, 0.67, 0.62, 0.62,
            0.80, 0.84, 0.84, 0.76, 0.71, 0.73,
            0.69, 0.64, 0.64, 0.67, 0.56, 0.51
        ],

        "Fear": [
            0.60, 0.59, 0.59, 0.66,
            0.00, 0.04, 0.09, 0.09, 0.04, 0.09,
            0.02, 0.18, 0.42, 0.31, 0.31, 0.40,
            0.00, 0.07, 0.11, 0.22, 0.18, 0.11,
            0.00, 0.09, 0.07, 0.09, 0.04, 0.04
        ],

        "Happy": [
            0.82, 0.87, 0.79, 0.86,
            0.91, 0.84, 0.80, 0.82, 0.84, 0.82,
            0.91, 0.80, 0.82, 0.76, 0.78, 0.78,
            0.96, 0.82, 0.89, 0.87, 0.76, 0.78,
            0.89, 0.84, 0.91, 0.82, 0.80, 0.87
        ],

        "Love": [
            0.87, 0.81, 0.85, 0.82,
            0.60, 0.67, 0.67, 0.73, 0.67, 0.69,
            0.38, 0.71, 0.73, 0.78, 0.73, 0.73,
            0.02, 0.31, 0.27, 0.33, 0.60, 0.47,
            0.40, 0.62, 0.33, 0.44, 0.58, 0.51
        ],

        "Sadness": [
            0.64, 0.66, 0.73, 0.72,
            0.76, 0.82, 0.82, 0.87, 0.89, 0.96,
            0.93, 0.78, 0.53, 0.64, 0.73, 0.67,
            0.80, 0.58, 0.62, 0.60, 0.62, 0.69,
            0.84, 0.93, 0.91, 0.91, 0.93, 0.93
        ],

        "Recall Macro Avg": [
            0.72, 0.71, 0.71, 0.75,
            0.61, 0.62, 0.61, 0.64, 0.63, 0.64,
            0.57, 0.63, 0.63, 0.63, 0.64, 0.64,
            0.52, 0.52, 0.55, 0.56, 0.57, 0.56,
            0.56, 0.63, 0.57, 0.59, 0.58, 0.57
        ]
    }
    df_recall = pd.DataFrame(recall_data)
    styled_df = df_recall.style.apply(highlight_max, subset=[
        "Anger", "Fear", "Happy", "Love", "Sadness", "Recall Macro Avg"
    ])
    st.dataframe(styled_df, use_container_width=True)
    st.divider()

    st.subheader("📈 Perbandingan F1 Score per Label")
    f1_data = {
        "Model": [
            "DistilBERT", "RoBERTa", "mBERT", "IndoBERT",
            "Gemini 2.5 Flash (0-shot)", "Gemini 2.5 Flash (5-shot)", "Gemini 2.5 Flash (10-shot)",
            "Gemini 2.5 Flash (15-shot)", "Gemini 2.5 Flash (20-shot)", "Gemini 2.5 Flash (25-shot)",
            "GPT-5.1 (0-shot)", "GPT-5.1 (5-shot)", "GPT-5.1 (10-shot)", "GPT-5.1 (15-shot)",
            "GPT-5.1 (20-shot)", "GPT-5.1 (25-shot)",
            "Llama3.1-8B (0-shot)", "Llama3.1-8B (5-shot)", "Llama3.1-8B (10-shot)", "Llama3.1-8B (15-shot)",
            "Llama3.1-8B (20-shot)", "Llama3.1-8B (25-shot)",
            "Gemma3-12B (0-shot)", "Gemma3-12B (5-shot)", "Gemma3-12B (10-shot)", 
            "Gemma3-12B (15-shot)", "Gemma3-12B (20-shot)", "Gemma3-12B (25-shot)"
        ],

        "Anger": [
            0.67, 0.68, 0.65, 0.73,
            0.64, 0.66, 0.63, 0.67, 0.65, 0.67,
            0.60, 0.62, 0.63, 0.63, 0.63, 0.66,
            0.66, 0.63, 0.67, 0.66, 0.61, 0.67,
            0.60, 0.67, 0.66, 0.65, 0.60, 0.57
        ],

        "Fear": [
            0.58, 0.57, 0.58, 0.63,
            0.00, 0.08, 0.16, 0.16, 0.08, 0.16,
            0.04, 0.26, 0.45, 0.37, 0.38, 0.44,
            0.00, 0.11, 0.17, 0.29, 0.25, 0.16,
            0.00, 0.15, 0.12, 0.15, 0.08, 0.08
        ],

        "Happy": [
            0.82, 0.81, 0.79, 0.83,
            0.80, 0.78, 0.76, 0.80, 0.78, 0.78,
            0.73, 0.77, 0.79, 0.76, 0.76, 0.76,
            0.67, 0.66, 0.69, 0.69, 0.71, 0.69,
            0.73, 0.78, 0.71, 0.70, 0.73, 0.74
        ],

        "Love": [
            0.86, 0.83, 0.84, 0.86,
            0.73, 0.76, 0.74, 0.79, 0.75, 0.76,
            0.53, 0.76, 0.79, 0.79, 0.77, 0.77,
            0.04, 0.42, 0.39, 0.46, 0.66, 0.56,
            0.55, 0.72, 0.48, 0.56, 0.68, 0.65
        ],

        "Sadness": [
            0.65, 0.67, 0.70, 0.70,
            0.59, 0.60, 0.60, 0.62, 0.62, 0.62,
            0.61, 0.61, 0.50, 0.57, 0.61, 0.57,
            0.60, 0.54, 0.56, 0.53, 0.55, 0.54,
            0.61, 0.63, 0.62, 0.64, 0.60, 0.59
        ],

        "F1 Score Macro Avg": [
            0.72, 0.71, 0.71, 0.75,
            0.55, 0.58, 0.58, 0.60, 0.58, 0.60,
            0.50, 0.60, 0.63, 0.62, 0.63, 0.64,
            0.39, 0.47, 0.50, 0.53, 0.55, 0.53,
            0.50, 0.59, 0.52, 0.54, 0.54, 0.53
        ]
    }
    df_f1 = pd.DataFrame(f1_data)
    styled_df = df_f1.style.apply(highlight_max, subset=[
        "Anger", "Fear", "Happy", "Love", "Sadness", "F1 Score Macro Avg"
    ])
    st.dataframe(styled_df, use_container_width=True)
    st.divider()

    st.subheader("🏆 Model Ranking by Accuracy")
    st.image("https://raw.githubusercontent.com/RasyadBima15/Multiclass-Emotion-Classification-on-Indonesian-Product-Reviews/main/streamlit/assets/model ranking by accuracy.png", caption="Model Ranking by Accuracy")
    st.divider()

    st.subheader("⚡ Model Ranking by Latency")
    st.image("https://raw.githubusercontent.com/RasyadBima15/Multiclass-Emotion-Classification-on-Indonesian-Product-Reviews/main/streamlit/assets/model ranking by latency.png", caption="Model Ranking by Latency")
    st.divider()

    st.subheader("📦 Model Ranking by Throughput")
    st.image("https://raw.githubusercontent.com/RasyadBima15/Multiclass-Emotion-Classification-on-Indonesian-Product-Reviews/main/streamlit/assets/model ranking by throughput.png", caption="Model Ranking by Throughput")

    st.divider()

    # ===========================
    # TRAINING TIME COMPARISON
    # ===========================
    st.subheader("⏱ Perbandingan Waktu Pelatihan Fine-Tuning BERT")

    st.image("https://raw.githubusercontent.com/RasyadBima15/Multiclass-Emotion-Classification-on-Indonesian-Product-Reviews/main/streamlit/assets/time training.png", caption="Perbandingan waktu pelatihan per model")

    st.divider()

    # =============================
    # LOSS & F1 SCORE PER EPOCH
    # =============================
    st.subheader("📉 Grafik Loss & F1-Score per Epoch Fine-Tuning BERT")

    st.image("https://raw.githubusercontent.com/RasyadBima15/Multiclass-Emotion-Classification-on-Indonesian-Product-Reviews/main/streamlit/assets/loss indobert.png", caption="Loss & F1-Score per Epoch - IndoBERT")
    st.image("https://raw.githubusercontent.com/RasyadBima15/Multiclass-Emotion-Classification-on-Indonesian-Product-Reviews/main/streamlit/assets/loss roberta.png", caption="Loss & F1-Score per Epoch - RoBERTa")
    st.image("https://raw.githubusercontent.com/RasyadBima15/Multiclass-Emotion-Classification-on-Indonesian-Product-Reviews/main/streamlit/assets/loss distilbert.png", caption="Loss & F1-Score per Epoch - DistilBERT")
    st.image("https://raw.githubusercontent.com/RasyadBima15/Multiclass-Emotion-Classification-on-Indonesian-Product-Reviews/main/streamlit/assets/loss mbert.png", caption="Loss & F1-Score per Epoch - mBERT")

def predict_single_page():
    """Halaman Prediksi Satu Teks."""
    st.title("📝 Prediksi Teks Tunggal")
    st.markdown("Halaman ini memungkinkan prediksi emosi untuk satu ulasan secara langsung menggunakan **IndoBERT**.")

    # Cek apakah model berhasil dimuat
    if model is None:
        st.error("Tidak dapat melanjutkan karena model gagal dimuat.")
        return

    # Area input teks
    text_input = st.text_area(
        "Masukkan teks ulasan yang ingin Anda prediksi emosinya:",
        placeholder="Contoh: Saya sangat suka dengan produk ini!",
        height=150
    )

    # Tombol Prediksi
    if st.button("Prediksi Emosi"):
        if text_input:
            with st.spinner('Menganalisis emosi...'):
                # Panggil fungsi prediksi
                predicted_label, probabilities = predict_emotion(
                    text_input, 
                    tokenizer, 
                    model
                )
            
            # Tampilkan Hasil
            st.success("✅ Prediksi Selesai!")
            # Map warna berdasarkan emosi
            COLOR_MAP = {
                "Anger": "#FF6B6B",
                "Fear": "#CBA3FF",
                "Happy": "#FFD93D",
                "Love": "#FF6EC7",
                "Sad": "#6BCBFF"
            }

            # Ambil warna sesuai prediksi
            color = COLOR_MAP.get(predicted_label, "#FFFFFF")  # default putih kalau label unknown

            st.markdown(
                f"**Emosi yang Diprediksi:** <span style='color:{color}; font-size:20px; font-weight:bold;'>{predicted_label}</span>",
                unsafe_allow_html=True
            )
            
            # Detail Probabilitas
            st.subheader("📊 Detail Probabilitas")

            # Warna gelap untuk progress bar di background terang
            COLOR_MAP_DARK = {
                "Anger": "#D23434",     # dark red
                "Fear": "#8D2AD4",      # dark purple
                "Happy": "#F1BD38",     # goldenrod
                "Love": "#DC1694",      # medium violet red
                "Sad": "#56ABFF"    # dodger blue
            }

            for label, prob in probabilities.items():
                percent = prob * 100
                bar_color = COLOR_MAP_DARK.get(label, "#2F4F4F")  # default dark slate gray jika label unknown

                st.markdown(
                    f"""
                    <div style='margin-bottom:10px;'>
                        <b>{label}</b>: {percent:.2f}%
                        <div style='background: #E6E6E6; border-radius:5px; width:100%; height:20px;'>
                            <div style='width:{percent}%; background:{bar_color}; height:100%; border-radius:5px;'></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.warning("Mohon masukkan teks untuk melakukan prediksi.")

def predict_multiple_page():
    """Halaman Prediksi Banyak Teks."""
    st.title("📄 Predict Multiple Text")
    st.markdown("Halaman ini memungkinkan prediksi banyak ulasan sekaligus menggunakan file CSV.")
    # Placeholder untuk konten prediksi banyak teks
    st.info("Fitur ini sedang dalam pengembangan.")

# --- Fungsi Utama Navigasi ---
def main():
    """Mengatur Navigasi Sidebar."""

    # 1. Judul Sidebar yang Jelas
    st.sidebar.title("🧠 Multiclass Emotion Classifier for Indonesian E-Commerce Reviews")
    # st.sidebar.subheader("Pilih Halaman") # Opsional, jika judul sudah cukup

    menu = [
        "🏠 Home",
        "📊 Hasil Analisis Model",
        "✍️ Prediksi Teks Tunggal",
        "📁 Prediksi Teks Massal"
    ]

    # 2. Selectbox dengan Teks yang Lebih Baik
    selection = st.sidebar.selectbox("Pilih Halaman", menu)

    st.sidebar.markdown(f'<div style= height:160px </div>', unsafe_allow_html=True)

    # Menambahkan sedikit pemisah visual sebelum footer
    st.sidebar.markdown("---")

    st.sidebar.image("https://raw.githubusercontent.com/RasyadBima15/Multiclass-Emotion-Classification-on-Indonesian-Product-Reviews/main/streamlit/assets/Unhas.png")

    # 3. Copyright footer yang dipercantik
    st.sidebar.markdown(
        """
        <div style='text-align:center; font-size:12px; color: grey;'>
            © 2025 | Dibuat oleh <b>Rasyad Bimasatya</b>
        </div>
        """,
        unsafe_allow_html=True
    )

    # --- Pemuatan Halaman ---
    if selection == "🏠 Home":
        home_page()
    elif selection == "📊 Hasil Analisis Model":
        model_analysis_page()
    elif selection == "✍️ Prediksi Teks Tunggal":
        predict_single_page()
    elif selection == "📁 Prediksi Teks Massal":
        predict_multiple_page()

if __name__ == "__main__":
    main()