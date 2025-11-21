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

# --- Konfigurasi Aplikasi Streamlit ---
st.set_page_config(layout="wide", page_title="Tokopedia Review Analyzer")

# --- Variabel Global dan Definisi Kelas ---
EMOTION_CLASSES = ['Anger', 'Fear', 'Happy', 'Love', 'Sadness']

# List stop-words Bahasa Indonesia sederhana untuk Wordcloud/Top 10
# Idealnya, gunakan list stop-words dari NLTK atau Sastrawi yang lebih lengkap
STOP_WORDS_ID = set([
    'yang', 'dan', 'di', 'ini', 'itu', 'saya', 'tidak', 'ke', 'ya', 'ada', 
    'dari', 'untuk', 'dengan', 'sudah', 'tapi', 'sih', 'nya', 'banget', 
    'udah', 'pun', 'juga', 'kok', 'terlalu', 'jadi', 'buat', 'sangat', 'produk'
])

# --- MOCKING GEMINI 2.5 FLASH API ---

@st.cache_data
def mock_gemini_analyze_and_summarize(emotion_label, review_texts):
    """
    Simulasi pemanggilan Gemini 2.5 Flash untuk analisis dan peringkasan.
    Fungsi ini menggabungkan 'Analyze Emotion' (Poin) dan 'Summary' (Paragraf).
    """
    
    # Kumpulan ulasan per label di sini hanya digunakan untuk demonstrasi,
    # hasil analisis dan ringkasan disimulasikan berdasarkan label.

    if emotion_label == 'Happy':
        poin = [
            "Kualitas Produk Melampaui Ekspektasi Harga: Sering dipuji karena kualitasnya premium meskipun harganya terjangkau.",
            "Kecepatan dan Keamanan Logistik: Pengiriman yang super cepat dan pengepakan yang sangat aman menjadi sumber kepuasan tinggi.",
            "Layanan Penjual Ramah dan Responsif: Komunikasi penjual yang baik menciptakan pengalaman belanja yang mulus."
        ]
        summary = "Emosi Happy didorong oleh kombinasi sempurna antara nilai produk dan efisiensi layanan. Pelanggan merasa senang karena mereka tidak hanya mendapatkan barang berkualitas dengan harga yang bagus, tetapi juga didukung oleh proses pengiriman yang cepat dan penjual yang sangat kooperatif, membuat mereka merekomendasikan produk ini."
    
    elif emotion_label == 'Anger':
        poin = [
            "Kesalahan Fatal dalam Pemenuhan Pesanan: Sering terjadi kesalahan kirim (warna/varian) yang menyebabkan frustrasi tinggi.",
            "Penjual Tidak Responsif dan Kurang Akuntabilitas: Keluhan utama adalah penjual yang 'menghilang' setelah pembayaran, mengabaikan chat dan pertanyaan.",
            "Kualitas Produk Cacat dan Cepat Rusak: Produk gagal berfungsi atau rusak dalam waktu singkat setelah pemakaian, menunjukkan kualitas yang buruk."
        ]
        summary = "Kelompok ini menunjukkan kemarahan yang disebabkan oleh kegagalan mendasar dalam transaksi, mulai dari kesalahan pengiriman yang ceroboh hingga kualitas produk yang sangat buruk. Kemarahan diperparah oleh kurangnya respons dan tanggung jawab dari pihak penjual, yang menimbulkan rasa kerugian waktu dan uang bagi pelanggan."
    
    # ... Tambahkan simulasi untuk Fear, Love, dan Sadness ...
    
    elif emotion_label == 'Love':
        poin = [
            "Kecintaan terhadap Estetika dan Desain: Produk dianggap sangat indah, sempurna, dan menjadi favorit pribadi.",
            "Kualitas Jangka Panjang: Produk menunjukkan daya tahan luar biasa, membuat pelanggan merasa investasi yang dilakukan sangat berharga.",
            "Pembelian Berulang dan Loyalitas: Pelanggan menyatakan akan selalu membeli dari toko ini karena rasa percaya dan kepuasan yang mendalam."
        ]
        summary = "Emosi Love mencerminkan koneksi yang kuat antara pelanggan dan produk/toko. Sumber kepuasan utama datang dari kualitas produk yang melebihi standar dan estetika yang sangat disukai, mendorong loyalitas jangka panjang dan rekomendasi yang antusias."

    elif emotion_label == 'Sadness':
        poin = [
            "Kekecewaan Kecil (Tidak Sesuai Harapan): Produk hanya 'agak berbeda' dari foto atau deskripsi, menyebabkan sedikit kekecewaan.",
            "Kerusakan Minor saat Pengiriman: Barang sampai dengan penyok atau goresan kecil yang bisa dihindari, menimbulkan kesedihan.",
            "Kesempatan yang Hilang: Pengiriman lambat membuat produk terlambat untuk acara penting (misalnya kado), memicu rasa sedih."
        ]
        summary = "Kelompok Sadness menunjukkan kekecewaan tingkat menengah yang seringkali berhubungan dengan detail kecil yang luput atau masalah logistik. Emosi ini kurang intens dibandingkan Anger, namun tetap mencerminkan harapan yang tidak terpenuhi sepenuhnya, seperti kerusakan minor atau keterlambatan pengiriman."
    
    elif emotion_label == 'Fear':
        poin = [
            "Kekhawatiran Barang Hilang/Rusak: Ulasan yang berfokus pada kekhawatiran selama perjalanan paket dan meminta *double-packing*.",
            "Ketakutan Produk Palsu/KW: Kekhawatiran apakah produk yang diterima benar-benar original sesuai iklan, terutama untuk barang mahal.",
            "Kekhawatiran Privasi Data: Ulasan yang menyebutkan ketakutan data pribadi (alamat, nomor HP) disalahgunakan."
        ]
        summary = "Emosi Fear terutama berpusat pada risiko yang dirasakan dalam proses belanja online, seperti kekhawatiran terhadap keamanan paket dan keaslian produk. Pelanggan merasa lega jika barang tiba dengan aman, namun awalnya didorong oleh rasa cemas yang tinggi terhadap penipuan atau kerusakan logistik."

    else:
        poin, summary = ["Data analisis tidak tersedia untuk label ini"], "Ringkasan tidak tersedia."

    return poin, summary


# --- MOCKING DATA SCAPPING DAN KLASIFIKASI ---

@st.cache_data
def generate_mock_data():
    """Menghasilkan DataFrame tiruan seolah-olah sudah discrap dan diklasifikasi."""
    N = 500  # Jumlah ulasan tiruan
    np.random.seed(42)

    # Distribusi emosi tiruan (untuk demo)
    emotions = np.random.choice(
        EMOTION_CLASSES, 
        size=N, 
        p=[0.15, 0.05, 0.40, 0.25, 0.15] # 40% Happy, 25% Love, sisanya dibagi
    )
    
    # Fungsi untuk membuat teks ulasan tiruan
    def generate_review(emotion):
        if emotion == 'Happy':
            return np.random.choice([
                "Produknya bagus sekali, sesuai deskripsi. Pengiriman juga super cepat!",
                "Bahannya tebal dan nyaman dipakai. Harga sangat murah, puas!",
                "Penjual ramah, packaging aman. Anak saya suka banget.",
                "Mantap! Kualitas premium, cepat sampai. Reccomended seller!"
            ])
        elif emotion == 'Anger':
            return np.random.choice([
                "BARANG CACAT! Solnya copot setelah sehari. Penjual tidak mau ganti rugi.",
                "SALAH KIRIM WARNA! Pesan hijau datang kuning. Penjual tidak responsif sama sekali.",
                "Sudah seminggu status tidak berubah. Sangat kesal! Pelayanan terburuk.",
                "Produk palsu. Saya minta uang kembali, jangan beli di toko ini!"
            ])
        elif emotion == 'Love':
            return np.random.choice([
                "Cinta banget sama desainnya! Ini sudah pembelian ke-3 saya. Selalu memuaskan.",
                "Sempurna! Produk favorit baru saya. Tidak ada tandingan kualitasnya.",
                "Aku terharu, barangnya melebihi impianku. Terbaik di Tokopedia!",
                "Sangat berharga! Investasi terbaik yang pernah saya lakukan. Awesome!"
            ])
        elif emotion == 'Sadness':
            return np.random.choice([
                "Agak sedih, ternyata ukurannya sedikit kekecilan dari perkiraan saya.",
                "Datang terlambat untuk hadiah ulang tahun. Lumayan kecewa.",
                "Ada goresan kecil di body, sedikit mengurangi nilai estetikanya.",
                "Saya kira bahannya lebih tebal. Yah, lumayan lah."
            ])
        elif emotion == 'Fear':
            return np.random.choice([
                "Semoga aman sampai rumah, saya takut barang pecah karena ini barang pecah belah.",
                "Packingnya kurang tebal. Semoga ini bukan barang palsu. Was-was.",
                "Saya cemas pengiriman hilang. Seller tolong diproses cepat ya.",
                "Tolong jangan sampai data alamat saya bocor ya, agak takut."
            ])
        return "Ulasan netral."

    df = pd.DataFrame({
        'Review': [generate_review(e) for e in emotions],
        'Emotion': emotions,
        'Timestamp': pd.to_datetime(pd.to_datetime('2024-01-01') + pd.to_timedelta(np.random.randint(0, 365, N), unit='D')).strftime('%Y-%m-%d %H:%M:%S')
    })
    
    return df

# --- FUNGSI UTILITY UNTUK ANALISIS TEKS ---

@st.cache_data
def get_word_frequency(texts, top_n=10):
    """Menghitung frekuensi kata teratas setelah stop-word removal."""
    all_words = []
    for text in texts:
        # Hapus non-alfabetik, tokenize, dan ubah ke lowercase
        words = re.findall(r'\b[a-z]{2,}\b', text.lower())
        all_words.extend([word for word in words if word not in STOP_WORDS_ID])
    
    return Counter(all_words).most_common(top_n)

@st.cache_data
def generate_wordcloud_image(texts):
    """Membuat objek WordCloud."""
    text_combined = " ".join(texts)
    wc = WordCloud(
        width=800, height=400, 
        background_color="white", 
        stopwords=STOP_WORDS_ID, 
        max_words=100
    ).generate(text_combined)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis("off")
    plt.tight_layout(pad=0)
    return fig

@st.cache_data
def convert_df_to_csv(df):
    """Konversi DataFrame ke CSV string untuk diunduh."""
    # Pastikan hanya kolom penting yang diekspor
    df_export = df[['Review', 'Emotion', 'Timestamp']] 
    return df_export.to_csv(index=False).encode('utf-8')

# --- DEFINISI HALAMAN APLIKASI ---

def home_page():
    """Halaman Profil Singkat Aplikasi."""
    st.title("Tokopedia Review Sentiment Analyzer 📈")
    st.subheader("Solusi Cepat untuk Memahami Emosi Pelanggan Anda")
    
    st.markdown("""
        Selamat datang di alat analisis ulasan produk berbasis Gemini 2.5 Flash.
        Aplikasi ini dirancang untuk mengubah data ulasan mentah dari Tokopedia menjadi wawasan bisnis yang terstruktur.
        
        ### Fitur Utama
        
        1.  **Klasifikasi Emosi Multi-kelas:** Mengidentifikasi 5 emosi spesifik: **Anger (Marah), Fear (Takut), Happy (Senang), Love (Cinta), dan Sadness (Sedih).**
        2.  **Visualisasi Data:** Distribusi emosi, Wordcloud, dan Top 10 Kata Kunci.
        3.  **Analisis Kualitatif (Didukung LLM):** Gemini 2.5 Flash meringkas penyebab utama setiap emosi.
        
        ### Bagaimana Cara Kerjanya?
        
        1.  Anda melakukan *scrapping* ulasan dari produk Tokopedia (simulasi data dilakukan di sini).
        2.  Setiap ulasan diklasifikasikan emosinya.
        3.  Kelompok ulasan per emosi dikumpulkan, dan Gemini 2.5 Flash (simulasi) menganalisis dan membuat ringkasan naratif tentang apa yang dibicarakan dalam kelompok tersebut.
        
        Silakan navigasi ke **'Scrapping & Analyzer'** di sidebar untuk memulai analisis.
    """)
    st.image("https://placehold.co/800x300/e0f2f1/004d40?text=Dashboard+Visualisasi+Contoh", caption="Ilustrasi Visualisasi Hasil Analisis")

def analyzer_page():
    """Halaman Utama Scrapping dan Analisis."""
    st.title("Scrapping & Analyzer Produk Tokopedia")
    st.warning("⚠️ **CATATAN SIMULASI:** Proses *scrapping* dan pemanggilan API Gemini disimulasikan menggunakan data dan logika tiruan (mock data) untuk menampilkan semua fitur analisis.")
    
    # --- Input Simulasi ---
    st.header("1. Input Data Ulasan")
    url_input = st.text_input("Masukkan URL Produk Tokopedia", "https://www.tokopedia.com/contoh-produk-xyz", disabled=True)
    
    if st.button("Mulai Analisis Data (Simulasi)", type="primary"):
        # Memuat data (Simulasi Scrapping & Klasifikasi)
        with st.spinner('Mengumpulkan ulasan dan mengklasifikasi emosi dengan Gemini 2.5 Flash...'):
            time.sleep(1) # Simulasikan loading time
            df_final = generate_mock_data()
        
        st.session_state['df_analyzed'] = df_final
        st.success(f"✅ Analisis Selesai! {len(df_final)} ulasan berhasil diproses.")

    if 'df_analyzed' in st.session_state:
        df_final = st.session_state['df_analyzed']
        
        st.markdown("---")
        st.header("2. Hasil Analisis Emosi Komprehensif")
        
        # --- I. Statistik & Visualisasi ---
        
        # 1. Distribusi Kelas
        st.subheader("2.1. Distribusi Kelas Emosi")
        distribusi = df_final['Emotion'].value_counts().reset_index()
        distribusi.columns = ['Emosi', 'Jumlah Ulasan']
        
        fig_dist = px.bar(
            distribusi, 
            x='Emosi', 
            y='Jumlah Ulasan', 
            title='Distribusi Jumlah Ulasan Berdasarkan Emosi',
            color='Emosi',
            color_discrete_map={
                'Happy': '#4CAF50', 'Love': '#FFC107', 'Sadness': '#2196F3', 
                'Anger': '#F44336', 'Fear': '#9E9E9E'
            },
            template='plotly_white'
        )
        st.plotly_chart(fig_dist, use_container_width=True)
        
        st.markdown("---")
        
        # 2 & 3. Wordcloud & Top 10 Words
        st.subheader("2.2. Analisis Kata Kunci (Per Emosi)")
        
        tabs = st.tabs(EMOTION_CLASSES)
        
        for i, emotion in enumerate(EMOTION_CLASSES):
            with tabs[i]:
                st.markdown(f"#### Analisis Kata Kunci: **{emotion.upper()}**")
                
                # Filter ulasan untuk emosi saat ini
                texts = df_final[df_final['Emotion'] == emotion]['Review'].tolist()
                
                if not texts:
                    st.info(f"Tidak ada ulasan yang diklasifikasikan sebagai {emotion}.")
                    continue
                
                col_wordcloud, col_top10 = st.columns([2, 1])

                # Wordcloud
                with col_wordcloud:
                    st.markdown("##### Wordcloud")
                    try:
                        fig_wc = generate_wordcloud_image(texts)
                        st.pyplot(fig_wc)
                    except Exception as e:
                        st.error(f"Gagal membuat Wordcloud: {e}")

                # Top 10 Words
                with col_top10:
                    st.markdown("##### Top 10 Kata Kunci")
                    top_words = get_word_frequency(texts, 10)
                    df_top = pd.DataFrame(top_words, columns=['Kata', 'Frekuensi'])
                    st.dataframe(df_top, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # --- II. Analisis Kualitatif LLM ---
        st.subheader("2.3. Analisis dan Ringkasan Kualitatif (Powered by Gemini 2.5 Flash)")
        
        # Loop untuk analisis LLM per label
        analysis_cols = st.columns(len(EMOTION_CLASSES))
        
        for i, emotion in enumerate(EMOTION_CLASSES):
            with analysis_cols[i]:
                with st.container(border=True):
                    st.markdown(f"**{emotion.upper()}**", unsafe_allow_html=True)
                    
                    # Panggil simulasi LLM untuk mendapatkan Poin & Ringkasan
                    poin_analisis, ringkasan_paragraf = mock_gemini_analyze_and_summarize(
                        emotion, 
                        df_final[df_final['Emotion'] == emotion]['Review'].tolist()
                    )
                    
                    st.markdown("##### Poin Analisis Kunci:")
                    for poin in poin_analisis:
                        st.markdown(f"- {poin}")
                        
                    st.markdown("##### Ringkasan:")
                    st.caption(ringkasan_paragraf)
        
        st.markdown("---")

        # --- III. Data & Export ---
        
        # 6. Tampilkan 10 Sample Data
        st.header("3. Sampel Data Mentah & Hasil Klasifikasi")
        st.write("Menampilkan 10 baris acak dari seluruh ulasan yang telah diproses (Review, Emotion, Timestamp).")
        
        sample_data = df_final[['Review', 'Emotion', 'Timestamp']].sample(min(10, len(df_final)))
        st.dataframe(sample_data, use_container_width=True)

        # 7. Download Hasil File CSV
        st.markdown("---")
        st.header("4. Ekspor Data Lengkap")
        
        csv_data = convert_df_to_csv(df_final)

        st.download_button(
            label="📥 Download Hasil Analisis Lengkap (.CSV)",
            data=csv_data,
            file_name='tokopedia_review_analysis_gemini.csv',
            mime='text/csv',
            type="secondary"
        )
        

# --- Fungsi Utama Navigasi ---
def main():
    """Mengatur Navigasi Sidebar."""
    
    st.sidebar.title("Navigasi Aplikasi")
    selection = st.sidebar.radio("Pilih Halaman", ["Home (Profil)", "Scrapping & Analyzer"])
    
    if selection == "Home (Profil)":
        home_page()
    elif selection == "Scrapping & Analyzer":
        analyzer_page()

if __name__ == "__main__":
    # Setup untuk matplotlib (agar wordcloud tidak overlap)
    plt.rcParams['figure.dpi'] = 100
    main()