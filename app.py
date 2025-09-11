# app.py
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

    # (Dihilangkan) Pemakaian labels/descriptions manual
    return lda, dictionary, bigram, trigram, stopw

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
    """Gauge chart untuk confidence level (angka selalu center)."""
    pct = top_score * 100.0

    fig = go.Figure(go.Indicator(
        mode="gauge",               
        value=pct,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Tingkat Kepercayaan (%)", 'font': {'size': 16}},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "black"},
            'steps': [
                {'range': [0, 25],  'color': "red"},
                {'range': [25, 50], 'color': "orange"},
                {'range': [50, 75], 'color': "yellow"},
                {'range': [75, 100],'color': "green"},
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': pct
            }
        }
    ))

    # Tambahkan angka di tengah, selalu center relatif ke kanvas
    fig.add_annotation(
        x=0.5, y=0.1, xref="paper", yref="paper",
        text=f"{pct:.1f}%",
        showarrow=False,
        font=dict(size=60, color="white")  # ubah warna sesuai tema Anda
    )

    fig.update_layout(
        height=300,
        margin=dict(t=50, b=0, l=0, r=0)
    )
    return fig


def create_topic_distribution_chart(topic_dist, lda, top_n=8):
    """Interactive bar chart untuk distribusi topik"""
    top_topics = topic_dist[:top_n]
    
    # TAMPILKAN 1-BASED
    topic_labels = [f"Topik {tid+1}" for tid, _ in top_topics]
    scores = [score for _, score in top_topics]
    
    # Warna berbeda untuk top 3
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1'] + ['#96CEB4'] * (len(scores) - 3)
    
    fig = px.bar(
        x=scores, 
        y=topic_labels,
        orientation='h',
        title="Distribusi Topik Artikel",
        labels={'x': 'Probabilitas', 'y': 'Topik'},
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

def top_terms_list(lda, topic_id: int, topn: int = 10):
    return [w for (w, _) in lda.show_topic(topic_id, topn=topn)]

def top_terms_str(lda, topic_id: int, topn: int = 10):
    return ", ".join(top_terms_list(lda, topic_id, topn=topn))

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
        Menggunakan Topic Modeling (LDA) untuk mengekstraksi <em>topik laten</em> dari artikel berita
    </p>
</div>
""", unsafe_allow_html=True)

# Load model
try:
    lda, dictionary, bigram, trigram, stopw = load_artifacts()
    model_loaded = True
except Exception as e:
    st.error(f"❌ Gagal memuat model: {e}")
    st.stop()

# Sidebar info
with st.sidebar:
    # 1. HEADER PROFESIONAL
    st.title("Analisis Topik Berita dengan LDA")
    st.markdown("Oleh: **Wahyu Rahmadani**") 
    st.markdown("---")

    # 2. TENTANG SISTEM 
    st.markdown("### ℹ️ Tentang Sistem")
    st.info(
        "Aplikasi ini mengklasifikasikan artikel berita Indonesia ke dalam beberapa topik utama menggunakan model *Latent Dirichlet Allocation* (LDA)."
    )
    
    # 3. DETAIL MODEL (Gabungan Statistik & Performa)
    st.markdown("### 📈 Detail Model")
    
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
    4. **Output**: Menampilkan topik dominan & visualisasi.
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
        # yang Anda tahu akan memberikan hasil baik untuk topik tersebut.
        DEMO_URLS = {
            "Politik": "https://nasional.kompas.com/read/2025/08/22/12433401/kaesang-sebut-psi-harus-bawa-manfaat-besar-bukan-menjarah-rakyat",
            "Olahraga": "https://bola.kompas.com/read/2025/09/09/10281588/evaluasi-patrick-kluivert-usai-timnas-indonesia-imbang-lawan-lebanon",
            "Ekonomi": "https://finance.detik.com/berita-ekonomi-bisnis/d-8052246/ekonomi-tumbuh-5-12-tapi-penerimaan-pajak-malah-anjlok-kok-bisa"
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
                st.rerun() 
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
            label_visibility="collapsed"
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
    - Gunakan artikel minimal 3–5 paragraf
    - Pastikan teks dalam Bahasa Indonesia
    - Pilih artikel dari portal berita terpercaya
    
    **Catatan tentang sistem:**
    - Model *Latent Dirichlet Allocation (LDA)* bekerja secara unsupervised (tanpa label manual)
    - Hasil analisis berupa **Topik 1, Topik 2, ... Topik N**, masing-masing ditandai oleh kata kunci penting
    - Interpretasi makna setiap topik dilakukan berdasarkan kumpulan kata kunci yang muncul
    """)


# Analysis Results
if user_text:
    st.markdown("---")
    st.markdown(f"## 📊 Hasil Analisis: {article_title}")
    st.caption("Catatan: Penomoran topik di aplikasi ini menggunakan format **Topik 1..N** (bukan 0..N-1) agar selaras dengan tampilan pyLDAvis.")
    
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
    main_col1, main_col2 = st.columns(2)

    with main_col1:
        st.metric("📝 Jumlah Kata", len(user_text.split()))
        
        top_topic_id, top_score = topic_dist[0]
        # TAMPILKAN 1-BASED
        st.markdown("🎯 **Topik Dominan**")
        st.markdown(f"<p style='font-size: 1.5rem; font-weight: bold; margin: 0;'>Topik {top_topic_id+1}</p>", unsafe_allow_html=True)
        st.caption(f"Kata kunci: {top_terms_str(lda, top_topic_id, topn=10)}")

    with main_col2:
        st.metric("🔤 Token Diproses", len(tokens))
        st.metric("📈 Confidence", f"{top_score:.1%}")
    
    # Main results in tabs
    tab_names = [
        "🏆 Topik Dominan",
        "📊 Distribusi Topik",
        "☁️ Kata Kunci Artikel",
        "🔎 Visualisasi Inter-Topik (pyLDAvis)"
    ]
    tab1, tab2, tab3, tab4 = st.tabs(tab_names)
    
    with tab1:
        st.markdown("### 🥇 Top 3 Topik Artikel")
        
        for i, (topic_id, score) in enumerate(topic_dist[:3]):
            title = f"Topik {topic_id+1}"
            keywords = top_terms_str(lda, topic_id, topn=10)
            if i == 0:
                st.success(f"**#{i+1}. {title}** — kata kunci: {keywords}")
            elif i == 1:
                st.info(f"**#{i+1}. {title}** — kata kunci: {keywords}")
            else:
                st.warning(f"**#{i+1}. {title}** — kata kunci: {keywords}")
            st.metric("Skor", f"{score:.3f}")
            st.markdown("---")

        st.markdown("### 📈 Tingkat Kepercayaan Sistem")
        confidence_fig = create_confidence_gauge(topic_dist[0][1])
        st.plotly_chart(confidence_fig, width='stretch')
    
    with tab2:
        st.markdown("### 📊 Distribusi Semua Topik")
        
        dist_fig = create_topic_distribution_chart(topic_dist, lda)
        st.plotly_chart(dist_fig, width='stretch')
        
        with st.expander("📋 Lihat tabel detail"):
            df = pd.DataFrame([
                {
                    "Ranking": i+1,
                    "Topik": f"Topik {tid+1}",
                    "Probabilitas": f"{score:.4f}",
                    "Persentase": f"{score*100:.2f}%",
                    "Kata kunci (Top-10)": top_terms_str(lda, tid, topn=10)
                }
                for i, (tid, score) in enumerate(topic_dist)
            ])
            st.dataframe(df, width='stretch', hide_index=True)
    
    with tab3:
        st.markdown("### ☁️ Kata-kata Penting dalam Artikel")
        st.caption("Visualisasi kata-kata yang paling sering muncul dalam artikel Anda setelah preprocessing")
        
        wc = create_document_wordcloud(tokens)
        if wc:
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
            st.markdown("### 🔍 Visualisasi Inter-Topik (pyLDAvis)")
            
            st.info(
                """
                **Panduan Visualisasi:**
                - **Panel Kiri:** Peta jarak antar topik (klik lingkaran untuk memilih topik)
                - **Panel Kanan:** 30 kata paling relevan untuk topik yang dipilih
                - **Slider λ (Lambda):** Mengatur keseimbangan frekuensi vs eksklusivitas kata
                """
            )
            
            html_file_path = Path("pyldavis_visualization.html")

            if html_file_path.exists():
                with open(html_file_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                
                simple_wrapper = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <meta charset="utf-8">
                    <style>
                        body {{
                            margin: 0;
                            padding: 0;
                            overflow: auto;
                            background: white;
                        }}
                        
                        /* Biarkan pyLDAvis menggunakan ukuran default, hanya pastikan tidak terpotong */
                        #ldavis_el {{
                            margin: 0 !important;
                            padding: 0 !important;
                        }}
                    </style>
                </head>
                <body>
                    {html_content.split('<body>')[1].split('</body>')[0] if '<body>' in html_content else html_content}
                </body>
                </html>
                """
                
                # Tampilkan dengan ukuran yang cukup besar untuk menampung layout asli pyLDAvis
                components.html(simple_wrapper, height=750, scrolling=True)
                
            else:
                st.error(
                    "❌ **File visualisasi tidak ditemukan!**\n\n"
                    "File `pyldavis_visualization.html` belum ada. Pastikan Anda sudah:\n"
                    "1. Menjalankan script untuk generate model LDA\n"
                    "2. Membuat visualisasi pyLDAvis dengan `pyldavis.save_html()`"
                )

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem 0;">
    <p>Sistem Informasi Analisis Topik Berita Indonesia | Dikembangkan menggunakan LDA & Streamlit</p>
</div>
""", unsafe_allow_html=True)
