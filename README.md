
# Berita LDA Starter (Untuk Skripsi)

Paket kecil ini menyiapkan pipeline **pasca-scraping** untuk skripsi Anda:
- Preprocessing teks berita Indonesia (cleaning, stopwords, stemming, n-grams).
- Training **LDA + coherence grid search**.
- Inference topik untuk artikel baru.
- Artefak model siap diintegrasikan ke **Streamlit**.

## Struktur
```
berita-lda-starter/
├─ data/                  # taruh CSV scraped di sini
├─ models/                # artefak keluar di sini
└─ src/
   ├─ preprocessing.py
   ├─ train_lda.py
   └─ infer.py
```

## Cara pakai
```bash
cd berita-lda-starter/src

# 1) Train (coba beberapa K)
python train_lda.py --csv ../data/all_berita.csv --outdir ../models --k 10 15 20 25 --no_below 12 --no_above 0.5 --min_words 100

# 2) Lihat hasil
cat ../models/coherence_report.csv
cat ../models/best_meta.json

# 3) Inference cepat
python infer.py --model_dir ../models --text "Pemerintah mengumumkan kebijakan baru terkait pajak kendaraan bermotor..." --topn 3
```

> Pastikan environment sudah terpasang dependensi dari `requirements.txt`.
