import streamlit as st
import pandas as pd
import numpy as np
import time

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Clickbait Detector",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Mock Model / Fungsi Deteksi (Simulasi) ---
def detect_clickbait(text):
    """Simulasi pemanggilan model deteksi clickbait (BERT)."""
    # Prediksi simulasi: Jika judul panjang atau mengandung angka/tanda tanya, anggap Clickbait.
    is_clickbait = (len(text.split()) > 15) or ("?" in text) or any(c.isdigit() for c in text)
    
    if is_clickbait:
        prediction = "CLICKBAIT"
        confidence = np.random.uniform(0.70, 0.99)
    else:
        prediction = "NON-CLICKBAIT"
        confidence = np.random.uniform(0.80, 0.99)
        
    # Memilih model berdasarkan confidence (simulasi perbandingan)
    model_used = "BERT (Fine-Tuned)" if confidence > 0.85 else "Zero-Shot (LLM)"
    
    return prediction, confidence, model_used

# --- Halaman-Halaman Aplikasi ---
# --- Halaman-Halaman Aplikasi ---
def home_page():
    """Halaman Utama"""
    st.title("📰 Clickbait Detection App")

    st.markdown("""
    Aplikasi ini dikembangkan untuk menampilkan hasil penelitian serta mengimplementasikan model deteksi clickbait secara interaktif.  
    Penelitian ini berjudul:

    **"Analisis Performa Pendekatan Zero-Shot dan Few-Shot Learning pada Large Language Models (LLM) dengan Fine-Tuning Bidirectional Encoder Representations from Transformers (BERT): Studi Kasus Deteksi Clickbait Judul Berita Berbahasa Indonesia."**

    Penelitian berfokus pada perbandingan kinerja antara **pendekatan Zero-Shot dan Few-Shot Learning** menggunakan **Large Language Models (LLM)**,  
    serta penerapan **Fine-Tuning pada model BERT** untuk mendeteksi judul berita yang berpotensi bersifat **Clickbait**.
    """)

    st.subheader("📊 Dataset")
    st.markdown("""
    Model ini dilatih menggunakan dataset **CLICK-ID (2020)**, sebuah dataset untuk judul berita clickbait berbahasa Indonesia.  

    **Publikasi:** *CLICK-ID: A Novel Dataset for Indonesian Clickbait Headlines*  
    **Penulis:** Andika William, Yunita Sari  
    **Tahun:** 2020  
    **DOI:** [10.1016/j.dib.2020.106231](https://doi.org/10.1016/j.dib.2020.106231)
    """)

    st.subheader("🧩 Pendekatan Penelitian")
    st.markdown("""
    Penelitian ini membandingkan tiga pendekatan utama:
    1. **Fine-Tuning BERT** — Menyesuaikan bobot model terhadap dataset lokal.
    2. **Zero-Shot Learning (LLM)** — Menggunakan kemampuan generalisasi tanpa pelatihan tambahan.
    3. **Few-Shot Learning (LLM)** — Memberikan beberapa contoh label untuk meningkatkan performa prediksi.

    Proses evaluasi dilakukan menggunakan metrik **Accuracy**, **Precision**, **Recall**, **F1-Score**, **Latency**, dan **Throughput**.
    """)

    st.subheader("🏆 Hasil Utama Penelitian")
    st.success("""
    Model **DistilBERT (cahya/distilbert-base-indonesian)** yang telah di-*fine-tuning* 
    dipilih sebagai **model terbaik secara keseluruhan**, dengan keseimbangan optimal antara akurasi dan efisiensi komputasi.

    **📊 Ringkasan Hasil Evaluasi (Test Set):**
    - **F1-Score:** 0.91  
    - **Akurasi:** 0.91  
    - **Precision:** 0.91
    - **Recall:** 0.91   
    - **Rata-rata Latency:** 0.08 detik/sampel  
    - **Throughput:** 707 sampel/menit  
    - **Waktu Pelatihan:** 8.79 menit  

    Model ini menunjukkan performa serupa dengan **IndoBERT-base-p1**, namun dengan waktu pelatihan **4x lebih cepat** 
    dan efisiensi pemrosesan yang jauh lebih tinggi, menjadikannya pilihan ideal untuk implementasi praktis.

    Sebagai perbandingan, model berbasis **LLM (Zero-Shot/Few-Shot)** seperti *Llama3.1, Gemma3, Mistral, dan Deepseek-r1* 
    memiliki F1-Score lebih rendah (0.57–0.82) serta *latency* yang jauh lebih tinggi (5–9 detik/sampel).
    """)

    st.subheader("⚙️ Fitur Aplikasi")
    st.markdown("""
    Aplikasi ini memiliki beberapa fitur utama yang dapat diakses melalui **sidebar**:

    - 🔍 **Predict Single Title** — Deteksi satu judul berita dan klasifikasi otomatis sebagai *Clickbait* atau *Non-Clickbait*.  
    - 📂 **Batch Detection** — Unggah file CSV/Excel berisi kumpulan judul berita untuk deteksi clickbait secara massal.
    - 📈 **Result Analytics** — Menampilkan hasil analisis performa model seperti akurasi, F1-score, dll.  
    """)

    st.subheader("👨‍💻 Tentang Pengembang")
    st.markdown("""
    Aplikasi ini dikembangkan oleh **Rasyad Bimasatya**, mahasiswa Sistem Informasi Universitas Hasanuddin Angkatan 2022,  
    sebagai implementasi penelitian dalam bidang **Natural Language Processing (NLP)** dan **Large Language Models (LLM)**.
    """)

def detect_single_page():
    """Deteksi Clickbait untuk Input Teks Tunggal"""
    st.title("1️⃣ Deteksi Teks Tunggal")
    st.subheader("Uji Model Secara Langsung")

    title_input = st.text_area(
        "Masukkan Judul Berita atau Teks di Bawah:",
        placeholder="Contoh: Tak Disangka! Rahasia Sukses Jadi Miliarder Ternyata Sesimpel Ini, Wajib Coba!",
        height=100
    )

    if st.button("Analisis Judul", type="primary"):
        if title_input:
            with st.spinner('Model sedang menganalisis...'):
                time.sleep(1.5) # Simulasi waktu pemrosesan
                prediction, confidence, model_used = detect_clickbait(title_input)
            
            st.markdown("---")
            st.subheader("Hasil Prediksi")
            
            # Tampilan hasil
            if prediction == "CLICKBAIT":
                st.error(f"⚠️ **{prediction}**")
                st.balloons()
            else:
                st.success(f"✅ **{prediction}**")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Tingkat Keyakinan Model", f"{confidence:.2%}")
            with col2:
                st.metric("Model yang Digunakan (Simulasi)", model_used)

            st.markdown("---")
            st.caption("Catatan: Hasil ini adalah simulasi berdasarkan model prototipe. Model sebenarnya mungkin memberikan hasil yang berbeda.")
        else:
            st.warning("Mohon masukkan teks judul terlebih dahulu.")


def detect_batch_page():
    """Deteksi Clickbait Mode Batch (Upload File)"""
    st.title("📂 Deteksi Batch (CSV/Excel)")
    st.subheader("Proses Banyak Judul Sekaligus")

    uploaded_file = st.file_uploader(
        "Unggah file CSV atau Excel (.csv, .xlsx)",
        type=["csv", "xlsx"]
    )

    if uploaded_file is not None:
        try:
            # Memuat file
            if uploaded_file.name.endswith('.csv'):
                df_input = pd.read_csv(uploaded_file)
            else:
                df_input = pd.read_excel(uploaded_file)
            
            st.success(f"File **{uploaded_file.name}** berhasil diunggah.")
            st.dataframe(df_input.head(), use_container_width=True)
            
            # Asumsi kolom pertama adalah judul
            st.info(f"Menggunakan kolom pertama ({df_input.columns[0]}) sebagai teks judul untuk analisis.")

            if st.button("Mulai Deteksi Batch", type="primary"):
                with st.spinner('Memproses deteksi untuk semua judul, harap tunggu...'):
                    # Kolom untuk menyimpan hasil
                    df_input['Prediction'] = ''
                    df_input['Confidence'] = 0.0
                    df_input['Model_Used'] = ''
                    
                    # Simulasi pemrosesan baris demi baris
                    for index, row in df_input.iterrows():
                        title = row[df_input.columns[0]] # Ambil data dari kolom judul
                        if pd.notna(title):
                            prediction, confidence, model_used = detect_clickbait(str(title))
                            df_input.loc[index, 'Prediction'] = prediction
                            df_input.loc[index, 'Confidence'] = confidence
                            df_input.loc[index, 'Model_Used'] = model_used

                st.markdown("---")
                st.subheader("✅ Hasil Deteksi Batch")
                
                # Format tampilan
                df_result = df_input[[df_input.columns[0], 'Prediction', 'Confidence', 'Model_Used']]
                df_result['Confidence'] = df_result['Confidence'].map('{:.2%}'.format)

                st.dataframe(df_result, use_container_width=True, height=400)
                
                # Tombol unduh
                @st.cache_data
                def convert_df(df):
                    # Cache the conversion to prevent computation on every rerun
                    return df.to_csv(index=False).encode('utf-8')

                csv = convert_df(df_input)

                st.download_button(
                    label="⬇️ Unduh Hasil CSV",
                    data=csv,
                    file_name='hasil_deteksi_clickbait.csv',
                    mime='text/csv',
                )

        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses file: {e}")
            st.info("Pastikan file memiliki kolom teks yang valid.")


def result_analytics_page():
    """Visualisasi dan Perbandingan Performa Model"""
    st.title("📊 Result Analytics")
    st.subheader("Perbandingan Kinerja Model")
    
    st.markdown("""
        Bagian ini menyajikan metrik performa (Akurasi, F1-Score, Presisi, Recall) dari 
        model deteksi clickbait yang berbeda: **BERT (Fine-Tuned)** dan **Zero/Few-Shot (LLM)**.
        Visualisasi ini membantu dalam memilih model terbaik untuk implementasi produksi.
    """)

    # --- Data Mock untuk Simulasi Performa ---
    data = {
        'Model': ['BERT (Fine-Tuned)', 'Zero-Shot (LLM)'],
        'Accuracy': [0.92, 0.75],
        'F1-Score': [0.91, 0.73],
        'Precision': [0.93, 0.78],
        'Recall': [0.90, 0.70]
    }
    df_metrics = pd.DataFrame(data)

    st.markdown("---")
    st.subheader("Tabel Metrik Performa")
    st.dataframe(df_metrics.style.highlight_max(axis=0, subset=['Accuracy', 'F1-Score', 'Precision', 'Recall'], color='lightblue'), use_container_width=True)

    st.markdown("---")
    st.subheader("Visualisasi Metrik Utama")

    # Mengubah format data untuk Bar Chart (Long Format)
    df_long = pd.melt(df_metrics, id_vars=['Model'], var_name='Metric', value_name='Score')
    
    # Pilih metrik yang ingin ditampilkan
    metrics_to_plot = ['Accuracy', 'F1-Score', 'Precision', 'Recall']
    df_plot = df_long[df_long['Metric'].isin(metrics_to_plot)]

    # Streamlit Bar Chart
    st.bar_chart(df_plot, x='Metric', y='Score', color='Model')

    st.markdown("---")
    st.subheader("Kesimpulan Analisis (Simulasi)")
    st.info("""
        Dari data di atas, terlihat bahwa model **BERT (Fine-Tuned)** menunjukkan performa yang **jauh lebih unggul**
        di semua metrik utama dibandingkan dengan pendekatan Zero-Shot Learning. 
        Hal ini mengindikasikan bahwa *fine-tuning* pada dataset spesifik Indonesia 
        memberikan hasil klasifikasi yang lebih akurat dan terpercaya.
    """)


# --- Struktur Navigasi Utama ---

st.sidebar.title("🧭 Navigasi Aplikasi")
st.sidebar.markdown("Pilih halaman untuk menjelajahi fitur-fitur aplikasi deteksi clickbait:")

page = st.sidebar.radio(
    "Pilih Halaman:",
    [
        "🏠 Home",
        "🔍 Predict Single Title",
        "📂 Batch Detection",
        "📈 Result Analytics"
    ],
    index=0
)

if page == "🏠 Home":
    home_page()
elif page == "🔍 Predict Single Title":
    detect_single_page()
elif page == "📂 Batch Detection":
    detect_batch_page()
elif page == "📈 Result Analytics":
    result_analytics_page()

st.sidebar.markdown("---")
st.sidebar.caption("2025 © Rasyad Bimasatya")