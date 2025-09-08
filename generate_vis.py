# generate_vis.py
import pickle
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
from gensim.models import LdaModel
from gensim.corpora import Dictionary

# --- SESUAIKAN BAGIAN INI ---
MODEL_PATH = "models_tune/lda_K10.model"
DICT_PATH = "models_tune/dictionary.dict"
# Anda HARUS menyediakan path ke file corpus training Anda.
# Ini adalah data yang Anda gunakan untuk melatih model LDA.
# Contoh jika Anda menyimpannya dalam format pickle:
CORPUS_PATH = "models_tune/corpus.pkl" 
OUTPUT_PATH = "pyldavis_visualization.html"
# --- AKHIR BAGIAN PENYESUAIAN ---

print("Memuat Model, Dictionary, dan Corpus Training...")
try:
    lda_model = LdaModel.load(MODEL_PATH)
    dictionary = Dictionary.load(DICT_PATH)
    
    # Ganti baris ini sesuai dengan cara Anda menyimpan corpus
    with open(CORPUS_PATH, "rb") as f:
        bow_corpus = pickle.load(f)

except FileNotFoundError as e:
    print(f"Error: File tidak ditemukan - {e}")
    print("Pastikan path untuk model, dictionary, dan corpus sudah benar.")
    exit()

print("Menyiapkan data untuk visualisasi pyLDAvis (ini mungkin memakan waktu)...")
vis_data = gensimvis.prepare(lda_model, bow_corpus, dictionary, mds='mmds')

print(f"Menyimpan visualisasi ke file: {OUTPUT_PATH}")
pyLDAvis.save_html(vis_data, OUTPUT_PATH)

print("\nSelesai! Visualisasi telah berhasil dibuat.")
print(f"Letakkan file '{OUTPUT_PATH}' di direktori yang sama dengan 'app.py' Anda.")
