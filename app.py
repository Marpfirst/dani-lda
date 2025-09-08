# app.py - IMPROVED: Fokus User Experience untuk Sistem Informasi
import json, pickle, requests, sys
import streamlit as st
from pathlib import Path
from bs4 import BeautifulSoup
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import pandas as pd
import streamlit.components.v1 as components


# modul preprocessing di root yang sama
sys.path.append(str(Path(__file__).resolve().parent / "src"))
from preprocessing import build_stopwords, preprocess_document, apply_ngrams

# ====== KONFIG ======
MODEL_DIR = Path("models")
LABELS_PATH = MODEL_DIR / "topic_labels.json"
EXTRA_STOP_PATH = Path("data/custom_stopwords.txt")

# ====== LOAD ARTEFAK ======
@st.cache_resource
def load_artifacts():
    lda = LdaModel.load(str(MODEL_DIR / "lda_K10.model"))
    dictionary = Dictionary.load(str(MODEL_DIR / "dictionary.dict"))
    with open(MODEL_DIR / "bigram.phraser", "rb") as f:
        bigram = pickle.load(f)
    with open(MODEL_DIR / "trigram.phraser", "rb") as f:
        trigram = pickle.load(f)

    stopw = build_stopwords()
    if EXTRA_STOP_PATH.exists():
        extra = [w.strip().lower() for w in EXTRA_STOP_PATH.read_text(encoding="utf-8").splitlines() if w.strip()]
        stopw |= set(extra)

    # Load topic labels dan deskripsi
    labels = {str(t): f"Topik {t}" for t in range(lda.num_topics)}
    descriptions = {str(t): "Kategori berita yang diidentifikasi oleh sistem." for t in range(lda.num_topics)}
    
    if LABELS_PATH.exists():
        try:
            data = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                # Jika ada struktur: {"0": {"label": "...", "desc": "..."}}
                for k, v in data.items():
                    if isinstance(v, dict):
                        labels[k] = v.get("label", f"Topik {k}")
                        descriptions[k] = v.get("desc", "Kategori berita yang diidentifikasi oleh sistem.")
                    else:
                        labels[k] = str(v)
        except Exception:
            pass

    return lda, dictionary, bigram, trigram, stopw, labels, descriptions

def extract_text_from_url(url: str) -> str:
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    r = requests.get(url, timeout=15, headers=headers)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    
    # Hapus script/style
    for script in soup(["script", "style"]):
        script.decompose()
    
    # Ambil paragraf
    paras = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    text = "\n".join([t for t in paras if t and len(t) > 20])
    
    if not text:
        text = soup.get_text(" ", strip=True)
    
    return text

def infer(text: str, lda, dictionary, bigram, trigram, stopw):
    toks = preprocess_document(text, stopw)
    toks = apply_ngrams(toks, bigram, trigram)
    bow = dictionary.doc2bow(toks)
    
    # Get full distribution
    topic_dist = lda.get_document_topics(bow, minimum_probability=0)
    topic_dist = sorted(topic_dist, key=lambda x: -x[1])
    
    return topic_dist, toks

def create_confidence_gauge(top_score):
    """Gauge chart untuk confidence level"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = top_score * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Tingkat Kepercayaan (%)"},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "white"},   # jarum/penunjuk
            'steps': [
                {'range': [0, 25],  'color': "red"},
                {'range': [25, 50], 'color': "orange"},
                {'range': [50, 75], 'color': "yellow"},
                {'range': [75, 100],'color': "green"}],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': top_score * 100}}))
    
    fig.update_layout(height=300)
    return fig

def create_topic_distribution_chart(topic_dist, labels, top_n=8):
    """Interactive bar chart untuk distribusi topik"""
    top_topics = topic_dist[:top_n]
    
    topic_labels = [labels.get(str(tid), f"Topik {tid}") for tid, _ in top_topics]
    scores = [score for _, score in top_topics]
    
    # Warna berbeda untuk top 3
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1'] + ['#96CEB4'] * (len(scores) - 3)
    
    fig = px.bar(
        x=scores, 
        y=topic_labels,
        orientation='h',
        title="Distribusi Kategori Berita",
        labels={'x': 'Probabilitas', 'y': 'Kategori'},
        color=topic_labels,
        color_discrete_sequence=colors
    )
    
    fig.update_layout(
        height=400,
        showlegend=False,
        yaxis={'categoryorder': 'total ascending'}
    )
    
    # Tambahkan nilai di ujung bar
    for i, score in enumerate(scores):
        fig.add_annotation(
            x=score + 0.01,
            y=i,
            text=f"{score:.3f}",
            showarrow=False,
            font=dict(size=10)
        )
    
    return fig

def create_document_wordcloud(tokens, max_words=30):
    """Wordcloud dari dokumen user"""
    if not tokens or len(tokens) < 5:
        return None
    
    word_freq = Counter(tokens)
    top_words = dict(word_freq.most_common(max_words))
    
    if not top_words:
        return None
    
    wc = WordCloud(
        width=800, 
        height=400, 
        background_color="white",
        max_words=max_words,
        colormap="viridis",
        relative_scaling=0.5
    ).generate_from_frequencies(top_words)
    
    return wc

def is_too_short(tokens, min_tokens=30):
    return tokens is None or len(tokens) < min_tokens

# ====== UI ======
st.set_page_config(
    page_title="Sistem Analisis Topik Berita", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Header
st.markdown("""
<div style="text-align: center; padding: 2rem 0;">
    <h1>🔍 Sistem Analisis Topik Berita Indonesia</h1>
    <p style="font-size: 1.2rem; color: #666;">
        Menggunakan Machine Learning (LDA) untuk mengidentifikasi kategori berita secara otomatis
    </p>
</div>
""", unsafe_allow_html=True)

# Load model
try:
    lda, dictionary, bigram, trigram, stopw, labels, descriptions = load_artifacts()
    model_loaded = True
except Exception as e:
    st.error(f"❌ Gagal memuat model: {e}")
    st.stop()

# Sidebar info
with st.sidebar:
    # 1. HEADER PROFESIONAL
    st.title("Analisis Topik Berita dengan LDA") # Ganti dengan judul skripsi Anda
    st.markdown("Oleh: **Wahyu Rahmadani**") # Ganti dengan nama Anda
    st.markdown("---") # Garis pemisah bisa diganti dengan st.divider()

    # 2. TENTANG SISTEM (Dibuat lebih ringkas)
    st.markdown("### ℹ️ Tentang Sistem")
    st.info(
        "Aplikasi ini mengklasifikasikan artikel berita Indonesia ke dalam beberapa topik utama menggunakan model *Latent Dirichlet Allocation* (LDA)."
    )
    
    # 3. DETAIL MODEL (Gabungan Statistik & Performa)
    st.markdown("### 📈 Detail Model")
    
    # Gunakan 2 kolom untuk metrik utama agar hemat tempat
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Jumlah Topik", lda.num_topics)
    with col2:
        # Menambahkan format ribuan agar mudah dibaca
        st.metric("Ukuran Kosakata", f"{len(dictionary):,}")

    # Expander untuk metrik performa yang lebih teknis
    with st.expander("Lihat Metrik Performa Model"):
        st.markdown(
            """
            - **Coherence (c_v):** `0.453`
            - **Log Perplexity:** `-7.83`
            - **Topic Diversity:** `Tinggi`
            """
        )
        st.markdown("---")
        st.caption(
            """
            **Dataset Training:** 4.706 artikel berita dari 4 portal mayor Indonesia (Juni–Juli 2025).
            """
        )

    # 4. CARA KERJA (Tetap sama, sudah bagus)
    st.markdown("### 🎯 Cara Kerja")
    st.markdown("""
    1. **Input**: Masukkan URL artikel berita.
    2. **Ekstraksi**: Sistem mengambil teks dari URL.
    3. **Analisis**: Model LDA memprediksi distribusi topik.
    4. **Output**: Menampilkan kategori dominan & visualisasi.
    """)

    # 5. FOOTER (Opsional, tapi menambah kesan profesional)
    st.markdown("---")
    st.caption("Dibuat untuk Program Skripsi | September 2025")

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("## 📝 Input Artikel")
    
    # Input method
    input_method = st.radio(
        "Pilih metode input:",
        ["🔗 URL Artikel", "📄 Tempel Teks"],
        horizontal=True
    )
    
    user_text = ""
    article_title = ""
    
    if input_method == "🔗 URL Artikel":
        # --- BLOK INI YANG KITA MODIFIKASI ---

        # 1. Siapkan beberapa contoh URL Demo
        # PENTING: Ganti URL di bawah ini dengan URL artikel berita asli
        # yang Anda tahu akan memberikan hasil baik untuk kategori tersebut.
        DEMO_URLS = {
            "Politik": "https://news.detik.com/pemilu/d-7734827/respons-anies-prabowo-dan-ganjar-soal-mundurnya-mahfud-md-dari-kabinet",
            "Olahraga": "https://sport.detik.com/sepakbola/d-7737217/liverpool-hajar-chelsea-4-1",
            "Ekonomi": "https://finance.detik.com/moneter/d-7736486/sri-mulyani-tarik-utang-rp-57-5-t-di-januari-2024"
        }

        # Inisialisasi session state jika belum ada
        if "demo_url" not in st.session_state:
            st.session_state.demo_url = ""

        # Tampilkan beberapa tombol contoh
        st.markdown("👇 **Atau coba dengan contoh artikel:**")
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("Politik", width='stretch'):
                st.session_state.demo_url = DEMO_URLS["Politik"]
                st.rerun() # Perintahkan Streamlit untuk refresh dengan state baru
        with c2:
            if st.button("Olahraga", width='stretch'):
                st.session_state.demo_url = DEMO_URLS["Olahraga"]
                st.rerun()
        with c3:
            if st.button("Ekonomi", width='stretch'):
                st.session_state.demo_url = DEMO_URLS["Ekonomi"]
                st.rerun()

        # Kotak input URL yang nilainya terhubung dengan session state
        url = st.text_input(
            "Masukkan URL artikel berita:",
            placeholder="https://...",
            value=st.session_state.demo_url,
            key="url_input",
            label_visibility="collapsed" # Sembunyikan label karena sudah ada di atas
        )

        if st.button("🚀 Analisis Artikel", type="primary", width='stretch'):
            if not url:
                st.warning("⚠️ Silakan masukkan URL atau pilih salah satu contoh di atas!")
            else:
                try:
                    with st.spinner("🔄 Mengambil konten artikel..."):
                        user_text = extract_text_from_url(url)
                        article_title = url.split("/")[-1].replace("-", " ").title()
                    if len(user_text) < 100:
                        st.warning("⚠️ Artikel terlalu pendek untuk dianalisis dengan akurat.")
                    else:
                        st.success(f"✅ Berhasil mengambil artikel ({len(user_text)} karakter)")
                except Exception as e:
                    st.error(f"❌ Gagal mengambil artikel: {str(e)}")
        # --- AKHIR BLOK MODIFIKASI ---
    else:
        user_text = st.text_area(
            "Tempel teks artikel di sini:",
            height=200,
            placeholder="Masukkan teks artikel berita Indonesia di sini..."
        )
        if user_text:
            if len(user_text) < 100:
                st.warning("⚠️ Teks terlalu pendek. Minimal 100 karakter untuk hasil optimal.")
            else:
                article_title = "Artikel Manual"
                
with col2:
    st.markdown("## 💡 Tips")
    st.info("""
    **Untuk hasil terbaik:**
    - Artikel minimal 3-5 paragraf
    - Bahasa Indonesia
    - Dari portal berita terpercaya
    
    **Sistem dapat mengidentifikasi:**
    - Politik & Pemerintahan
    - Ekonomi & Bisnis  
    - Olahraga
    - Kriminal & Hukum
    - Dan 16 kategori lainnya
    """)

# Analysis Results
if user_text:
    st.markdown("---")
    st.markdown(f"## 📊 Hasil Analisis: {article_title}")
    
    with st.spinner("🤖 Menganalisis topik artikel..."):
        toks_tmp = preprocess_document(user_text, stopw)
        toks_tmp = apply_ngrams(toks_tmp, bigram, trigram)
        if is_too_short(toks_tmp):
            st.error("Teks terlalu pendek/bersih untuk pemetaan topik yang reliabel (token < 30). Tambahkan teks lain.")
            st.stop()

        bow_tmp = dictionary.doc2bow(toks_tmp)
        if len(bow_tmp) == 0:
            st.error("Tidak ada token yang cocok dengan kosakata model. Coba artikel lain atau kurangi stopwords khusus.")
            st.stop()

        topic_dist, tokens = infer(user_text, lda, dictionary, bigram, trigram, stopw)

# Basic stats
    # Menggunakan tata letak 2x2 untuk memberi lebih banyak ruang
    main_col1, main_col2 = st.columns(2)

    with main_col1:
        st.metric("📝 Jumlah Kata", len(user_text.split()))
        
        top_topic_id, top_score = topic_dist[0]
        # Menggunakan st.markdown untuk judul agar bisa wrapping jika perlu
        st.markdown("🎯 **Kategori Utama**")
        st.markdown(f"<p style='font-size: 1.5rem; font-weight: bold; margin: 0;'>{labels.get(str(top_topic_id), f'Topik {top_topic_id}')}</p>", unsafe_allow_html=True)


    with main_col2:
        st.metric("🔤 Token Diproses", len(tokens))
        st.metric("📈 Confidence", f"{top_score:.1%}")
    
    # Main results in tabs
    tab_names = [
        "🏆 Kategori Dominan", 
        "📊 Distribusi Lengkap", 
        "☁️ Kata Kunci Artikel", 
        "🔎 Visualisasi Interaktif (pyLDAvis)" # Tab baru
    ]
    tab1, tab2, tab3, tab4 = st.tabs(tab_names)
    
    with tab1:
        st.markdown("### 🥇 Top 3 Kategori Berita")
        
        for i, (topic_id, score) in enumerate(topic_dist[:3]):
            label = labels.get(str(topic_id), f"Topik {topic_id}")
            desc = descriptions.get(str(topic_id), "Kategori berita yang diidentifikasi oleh sistem.")
            
            if i == 0:
                st.success(f"**#{i+1}. {label}**")
            elif i == 1:
                st.info(f"**#{i+1}. {label}**")
            else:
                st.warning(f"**#{i+1}. {label}**")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                st.caption(desc)
            with col2:
                st.metric("Skor", f"{score:.3f}")
            
            st.markdown("---")
        
        st.markdown("### 📈 Tingkat Kepercayaan Sistem")
        confidence_fig = create_confidence_gauge(topic_dist[0][1])
        st.plotly_chart(confidence_fig, width='stretch')
    
    with tab2:
        st.markdown("### 📊 Distribusi Semua Kategori")
        
        dist_fig = create_topic_distribution_chart(topic_dist, labels)
        st.plotly_chart(dist_fig, width='stretch')
        
        with st.expander("📋 Lihat tabel detail"):
            df = pd.DataFrame([
                {
                    "Ranking": i+1,
                    "Kategori": labels.get(str(tid), f"Topik {tid}"),
                    "Probabilitas": f"{score:.4f}",
                    "Persentase": f"{score*100:.2f}%"
                }
                for i, (tid, score) in enumerate(topic_dist)
            ])
            st.dataframe(df, width='stretch', hide_index=True)
    
    with tab3:
        st.markdown("### ☁️ Kata-kata Penting dalam Artikel")
        st.caption("Visualisasi kata-kata yang paling sering muncul dalam artikel Anda setelah preprocessing")
        
        wc = create_document_wordcloud(tokens)
        if wc:
            # --- INI BARIS YANG DIPERBAIKI ---
            st.image(wc.to_array(), width='stretch')
            
            word_freq = Counter(tokens)
            top_words_df = pd.DataFrame([
                {"Kata": word, "Frekuensi": freq}
                for word, freq in word_freq.most_common(15)
            ])
            
            with st.expander("📝 15 kata teratas"):
                st.dataframe(top_words_df, width='stretch', hide_index=True)
        else:
            st.warning("Tidak dapat membuat wordcloud - artikel terlalu pendek.")
            
    with tab4:
        st.markdown("### Visualisasi Inter-Topik (pyLDAvis)")
        st.info(
            """
            Visualisasi ini memungkinkan Anda untuk menjelajahi hubungan antar topik.
            - **Lingkaran di Kiri:** Setiap lingkaran adalah satu topik. Ukuran menunjukkan seberapa umum topik tersebut. Jarak antar lingkaran menunjukkan seberapa mirip mereka.
            - **Bagan di Kanan:** Menunjukkan kata-kata yang paling relevan untuk topik yang dipilih.
            """
        )
        
        # Path ke file HTML yang sudah dibuat
        html_file_path = Path("pyldavis_visualization.html")

        if html_file_path.exists():
            with open(html_file_path, 'r', encoding='utf-8') as f:
                html_string = f.read()
                # Tampilkan HTML di dalam komponen Streamlit tanpa width tetap
                components.html(html_string, height=800, scrolling=True)
        else:
            st.warning(
                "File visualisasi pyLDAvis tidak ditemukan. "
                "Harap jalankan skrip `generate_vis.py` terlebih dahulu."
            )

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem 0;">
    <p>Sistem Informasi Analisis Topik Berita Indonesia | Dikembangkan menggunakan LDA & Streamlit</p>
</div>
""", unsafe_allow_html=True)
