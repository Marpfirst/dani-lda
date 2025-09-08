
# -*- coding: utf-8 -*-
"""
Inference utilities: predict top-N topics for a new article (raw text)
"""
import argparse, pickle, json
from pathlib import Path
from typing import List, Tuple

from gensim.models import LdaModel
from gensim.corpora import Dictionary

from preprocessing import build_stopwords, preprocess_document, apply_ngrams

def load_artifacts(model_dir: str):
    model_dir = Path(model_dir)
    lda = LdaModel.load(str(model_dir / "lda_best.model"))
    dictionary = Dictionary.load(str(model_dir / "dictionary.dict"))
    with open(model_dir / "bigram.phraser", "rb") as f:
        bigram_phraser = pickle.load(f)
    with open(model_dir / "trigram.phraser", "rb") as f:
        trigram_phraser = pickle.load(f)
    return lda, dictionary, bigram_phraser, trigram_phraser

def infer(text: str, model_dir: str, topn: int = 3):
    lda, dictionary, bigram_phraser, trigram_phraser = load_artifacts(model_dir)
    stopw = build_stopwords()
    toks = preprocess_document(text, stopw)
    toks = apply_ngrams(toks, bigram_phraser, trigram_phraser)
    bow = dictionary.doc2bow(toks)
    dist = sorted(lda.get_document_topics(bow), key=lambda x: -x[1])[:topn]
    return dist

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="models")
    ap.add_argument("--text", required=True, help="Raw article text to infer topics")
    ap.add_argument("--topn", type=int, default=3)
    args = ap.parse_args()
    result = infer(args.text, args.model_dir, topn=args.topn)
    print(json.dumps({"topics": result}, ensure_ascii=False))

if __name__ == "__main__":
    main()
