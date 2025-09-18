# -*- coding: utf-8 -*-
"""
Train LDA with coherence grid search (versi PROGRESS + SAMPLING + MULTIPROCESS).
Input: CSV with columns [portal, title, released, url, content, content_len]
Outputs: models/, coherence_report.csv

Contoh pemakaian:
  # uji cepat dengan 1000 artikel, lebih ringan
  python train_lda.py --csv ../data/all_berita.csv --outdir ../models_test \
    --k 10 15 --sample 1000 --passes 5 --iterations 200 --workers 4

  # full run (default lebih berat)
python src/train_lda.py --csv data/all_berita_sorted.csv --outdir models --k 5 10 15 20 --no_below 12 --no_above 0.5 --min_words 100 --workers 4

"""
import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple, Iterable
import time
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm

# import util preprocessing
from preprocessing import (
    build_stopwords,
    preprocess_document,
    build_phrasers,
    apply_ngrams,
)

def load_corpus(csv_path: str, min_words: int = 100) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # dedup by url
    if "url" in df.columns:
        df = df.drop_duplicates(subset=["url"]).copy()
    # filter by length if available
    if "content_len" in df.columns:
        df["content_len"] = df["content_len"].fillna(0)
        df = df[df["content_len"] >= min_words]
    # basic NA filter
    df = df.dropna(subset=["content"]).reset_index(drop=True)
    return df

def stratified_sample(df: pd.DataFrame, n: int, key: str = "portal") -> pd.DataFrame:
    if n is None or n <= 0 or len(df) <= n:
        return df
    if key in df.columns:
        # ambil proporsional per portal
        groups = df.groupby(key)
        sizes = (groups.size() / len(df) * n).round().astype(int)
        parts = []
        for g, sz in sizes.items():
            part = groups.get_group(g).sample(min(sz, len(groups.get_group(g))), random_state=42)
            parts.append(part)
        out = pd.concat(parts, ignore_index=True)
        # jika rounding kurang/lebih, sesuaikan
        if len(out) > n:
            out = out.sample(n, random_state=42)
        elif len(out) < n:
            extra = df.drop(out.index, errors="ignore")
            if len(extra) > 0:
                out = pd.concat([out, extra.sample(n - len(out), random_state=42)], ignore_index=True)
        return out.reset_index(drop=True)
    # fallback simple random
    return df.sample(n, random_state=42).reset_index(drop=True)

def make_dictionary_and_corpus(docs_tokens: List[List[str]], no_below=12, no_above=0.5, keep_n=100000):
    from gensim.corpora import Dictionary
    dictionary = Dictionary(docs_tokens)
    dictionary.filter_extremes(no_below=no_below, no_above=no_above, keep_n=keep_n)
    corpus = [dictionary.doc2bow(doc) for doc in docs_tokens]
    return dictionary, corpus

def train_lda(dictionary, corpus, num_topics=20, passes=10, iterations=1000, chunksize=2000, random_state=42):
    from gensim.models import LdaModel
    lda = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=random_state,
        chunksize=chunksize,
        passes=passes,
        iterations=iterations,
        alpha="asymmetric",
        eta="auto",
        eval_every=None,
    )
    return lda

def coherence_score(lda, texts, dictionary):
    from gensim.models.coherencemodel import CoherenceModel
    cm = CoherenceModel(model=lda, texts=texts, dictionary=dictionary, coherence="c_v")
    return cm.get_coherence()

def _preprocess_one(args):
    text, stopw = args
    return preprocess_document(text, stopw)

def preprocess_all(texts: Iterable[str], stopw, workers: int = 1) -> List[List[str]]:
    """
    Preprocess dengan progress bar. Jika workers>1, gunakan multiprocessing.
    Penting (Windows): panggilan ini harus terjadi di dalam if __name__ == '__main__'
    """
    texts = list(texts)
    if workers and workers > 1:
        import multiprocessing as mp
        # gunakan chunksize agar efisien
        chunk = max(16, len(texts) // (workers * 8) or 1)
        with mp.get_context("spawn").Pool(processes=workers) as pool:
            it = pool.imap(_preprocess_one, ((t, stopw) for t in texts), chunksize=chunk)
            return [doc for doc in tqdm(it, total=len(texts), desc="Preprocessing (mp)")]
    else:
        docs = []
        for t in tqdm(texts, desc="Preprocessing"):
            docs.append(preprocess_document(t, stopw))
        return docs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to CSV scraped news")
    ap.add_argument("--outdir", default="models", help="Where to save models and artifacts")
    ap.add_argument("--k", nargs="+", type=int, default=[10, 15, 20, 25], help="List of topic numbers to try")
    ap.add_argument("--no_below", type=int, default=12)
    ap.add_argument("--no_above", type=float, default=0.5)
    ap.add_argument("--min_words", type=int, default=100)
    ap.add_argument("--passes", type=int, default=10)
    ap.add_argument("--iterations", type=int, default=1000)
    ap.add_argument("--chunksize", type=int, default=2000)
    ap.add_argument("--sample", type=int, default=0, help="If >0, sample that many docs (stratified by portal if available)")
    ap.add_argument("--workers", type=int, default=1, help="Preprocess in parallel (Windows-safe).")
    args = ap.parse_args()

    t0_total = time.time()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("🔹 Loading corpus...")
    df = load_corpus(args.csv, min_words=args.min_words)
    print(f"   -> Loaded {len(df):,} articles")

    if args.sample and args.sample > 0:
        df = stratified_sample(df, args.sample, key="portal")
        print(f"   -> Stratified sample to {len(df):,} docs")

    print("🔹 Building stopwords...")
    stopw = build_stopwords()
    # Tambahkan stopwords custom dari file txt
    extra_path = Path(args.csv).parent.parent / "data" / "custom_stopwords.txt"
    if extra_path.exists():
        with open(extra_path, "r", encoding="utf-8") as f:
            extra = [w.strip().lower() for w in f if w.strip()]
        stopw |= set(extra)
        print(f"   -> Loaded {len(extra)} extra stopwords")

    print("🔹 Preprocessing documents...")
    t0 = time.time()
    # IMPORTANT for Windows multiprocessing: ensure this runs in __main__
    docs_tokens = preprocess_all(df["content"].tolist(), stopw, workers=args.workers)
    t1 = time.time()
    print(f"   -> Done in {t1 - t0:.1f}s")

    print("🔹 Building n-grams...")
    t0 = time.time()
    bigram_phraser, trigram_phraser = build_phrasers(docs_tokens, min_count=10, threshold=10.0)
    docs_tokens = [apply_ngrams(doc, bigram_phraser, trigram_phraser) for doc in tqdm(docs_tokens, desc="Apply n-grams")]
    t1 = time.time()
    print(f"   -> Done in {t1 - t0:.1f}s")

    print("🔹 Building dictionary & corpus...")
    dictionary, corpus = make_dictionary_and_corpus(
        docs_tokens, no_below=args.no_below, no_above=args.no_above
    )

    # Save artefact corpus
    with open(outdir / "processed_texts.pkl", "wb") as f:
        pickle.dump(docs_tokens, f)
    with open(outdir / "corpus.pkl", "wb") as f:
        pickle.dump(corpus, f)
    # Save dictionary & phrasers
    dictionary.save(str(outdir / "dictionary.dict"))
    with open(outdir / "bigram.phraser", "wb") as f:
        pickle.dump(bigram_phraser, f)
    with open(outdir / "trigram.phraser", "wb") as f:
        pickle.dump(trigram_phraser, f)

    results = []
    best_model = None
    best_cv = -np.inf
    best_k = None

    print("🔹 Training LDA (grid K):")
    for k in args.k:
        print(f"   🔸 K={k} …")
        t0 = time.time()
        lda = train_lda(
            dictionary,
            corpus,
            num_topics=k,
            passes=args.passes,
            iterations=args.iterations,
            chunksize=args.chunksize,
        )
        train_sec = time.time() - t0
        cv = coherence_score(lda, docs_tokens, dictionary)
        print(f"      -> coherence c_v={cv:.4f} | {train_sec:.1f}s")
        results.append({"K": k, "coherence_cv": cv, "train_sec": round(train_sec, 1)})
        lda.save(str(outdir / f"lda_K{k}.model"))
        if cv > best_cv:
            best_cv, best_k, best_model = cv, k, lda

    # Save coherence report
    rep = pd.DataFrame(results).sort_values("coherence_cv", ascending=False)
    rep.to_csv(outdir / "coherence_report.csv", index=False)
    print("✅ Saved:", outdir / "coherence_report.csv")

    # Save best alias
    if best_model is not None:
        best_model.save(str(outdir / "lda_best.model"))
        with open(outdir / "best_meta.json", "w", encoding="utf-8") as f:
            json.dump({"best_k": int(best_k), "best_cv": float(best_cv)}, f, ensure_ascii=False, indent=2)
        print(f"✅ Best model: K={best_k}, c_v={best_cv:.4f}")

    print(f"⏱️ Total time: {time.time() - t0_total:.1f}s")

if __name__ == "__main__":
    # Penting untuk multiprocessing di Windows
    main()
