# -*- coding: utf-8 -*-
"""
Suite Evaluasi Lengkap Model LDA untuk Skripsi
==============================================
Tool komprehensif untuk menganalisis dan mengevaluasi model LDA yang sudah dilatih.
Output: visualisasi, tabel, dan laporan siap untuk skripsi.

Usage:
    python evaluate.py --model_dir models_tune --data_dir data --output_dir outputs_thesis
"""

import argparse
import json
import pickle
import warnings
from pathlib import Path
from typing import List, Dict, Tuple, Any
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

from gensim.models import LdaModel, CoherenceModel
from gensim.corpora import Dictionary

# Optional imports (only used if present)
try:
    # If you have src/preprocessing.py or preprocessing.py available,
    # we can rebuild texts/corpus when not saved.
    import sys
    here = Path(__file__).resolve().parent
    if (here / "src").exists():
        sys.path.append(str(here / "src"))
    from preprocessing import build_stopwords, preprocess_document, apply_ngrams
    HAVE_PRE = True
except Exception:
    HAVE_PRE = False

# Suppress warnings for clean output
warnings.filterwarnings('ignore')

# Matplotlib/Seaborn style
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16


class ThesisLDAEvaluator:
    def __init__(self, model_dir: str, data_dir: str, output_dir: str):
        self.model_dir = Path(model_dir)
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (self.output_dir / "visualizations").mkdir(exist_ok=True)
        (self.output_dir / "visualizations" / "wordclouds").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "tables").mkdir(exist_ok=True)
        (self.output_dir / "reports").mkdir(exist_ok=True)

        self.results: Dict[str, Any] = {}
        self.topic_labels = self._load_topic_labels()

        print("🎓 Thesis LDA Evaluator initialized")
        print(f"   Model dir : {self.model_dir}")
        print(f"   Data dir  : {self.data_dir}")
        print(f"   Output dir: {self.output_dir}")

    # -------------------------------
    # Loading
    # -------------------------------
    def _load_topic_labels(self) -> Dict[str, str]:
        p = self.model_dir / "topic_labels.json"
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {}

    def load_artifacts(self):
        """Load model dan semua artifacts"""
        print("\n📥 Loading model artifacts...")

        try:
            self.lda = LdaModel.load(str(self.model_dir / "lda_best.model"))
            self.dictionary = Dictionary.load(str(self.model_dir / "dictionary.dict"))

            with open(self.model_dir / "bigram.phraser", "rb") as f:
                self.bigram_phraser = pickle.load(f)
            with open(self.model_dir / "trigram.phraser", "rb") as f:
                self.trigram_phraser = pickle.load(f)

            self.corpus = None
            self.texts = None

            corpus_path = self.model_dir / "corpus.pkl"
            if corpus_path.exists():
                with open(corpus_path, "rb") as f:
                    self.corpus = pickle.load(f)

            texts_path = self.model_dir / "processed_texts.pkl"
            if texts_path.exists():
                with open(texts_path, "rb") as f:
                    self.texts = pickle.load(f)

            coherence_path = self.model_dir / "coherence_report.csv"
            self.coherence_report = pd.read_csv(coherence_path) if coherence_path.exists() else None

            print(f"   ✅ Model loaded: {self.lda.num_topics} topics")
            print(f"   ✅ Dictionary: {len(self.dictionary)} words")
            print(f"   ✅ Corpus: {'✓' if self.corpus is not None else '✗'}")
            print(f"   ✅ Texts : {'✓' if self.texts  is not None else '✗'}")
            print(f"   ✅ Coherence report: {'✓' if self.coherence_report is not None else '✗'}")

        except Exception as e:
            raise Exception(f"Error loading artifacts: {e}")

    def load_dataset(self):
        """Load dataset asli"""
        print("\n📊 Loading original dataset...")
        csv_path = self.data_dir / "all_berita.csv"
        self.df = pd.read_csv(csv_path)

        # Basic date normalization for printing ranges
        if 'released' in self.df.columns:
            try:
                self.df['released'] = pd.to_datetime(self.df['released'])
            except Exception:
                pass

        print(f"   ✅ Dataset loaded: {len(self.df):,} articles")
        if 'portal' in self.df.columns:
            print(f"   📰 Portals: {self.df['portal'].nunique()}")
        if 'released' in self.df.columns:
            try:
                print(f"   📅 Date range: {self.df['released'].min()} - {self.df['released'].max()}")
            except Exception:
                pass

        self.results['dataset_stats'] = {
            'total_articles': int(len(self.df)),
            'portals': (self.df['portal'].value_counts().to_dict()
                        if 'portal' in self.df.columns else {}),
            'avg_content_length': float(self.df.get('content_len', pd.Series(dtype=float)).mean())
                if 'content_len' in self.df.columns else None,
            'median_content_length': float(self.df.get('content_len', pd.Series(dtype=float)).median())
                if 'content_len' in self.df.columns else None,
            'content_length_std': float(self.df.get('content_len', pd.Series(dtype=float)).std())
                if 'content_len' in self.df.columns else None,
            'date_range': {
                'start': str(self.df['released'].min()) if 'released' in self.df.columns else None,
                'end': str(self.df['released'].max()) if 'released' in self.df.columns else None
            }
        }

        # If corpus/texts are missing but we can rebuild, do it
        if (self.corpus is None or self.texts is None) and HAVE_PRE and 'content' in self.df.columns:
            print("   🔧 Rebuilding texts/corpus from raw dataset (this may take a while)...")
            stopw = build_stopwords()
            extra_path = self.data_dir / "custom_stopwords.txt"
            if extra_path.exists():
                extra = [w.strip().lower() for w in extra_path.read_text(encoding="utf-8").splitlines() if w.strip()]
                stopw |= set(extra)

            texts = []
            for txt in self.df['content'].fillna("").tolist():
                toks = preprocess_document(txt, stopw)
                toks = apply_ngrams(toks, self.bigram_phraser, self.trigram_phraser)
                texts.append(toks)
            self.texts = texts
            self.corpus = [self.dictionary.doc2bow(doc) for doc in self.texts]

            # simpan agar next run cepat
            with open(self.model_dir / "processed_texts.pkl", "wb") as f:
                pickle.dump(self.texts, f)
            with open(self.model_dir / "corpus.pkl", "wb") as f:
                pickle.dump(self.corpus, f)

            print("   ✅ Rebuilt texts & corpus, and saved to model_dir.")


    # -------------------------------
    # Evaluation
    # -------------------------------
    def evaluate_model_performance(self):
        """Evaluasi performa model dengan berbagai metrics"""
        print("\n🧮 Evaluating model performance...")

        performance: Dict[str, Any] = {}

        # 1) Coherence scores
        if self.texts is not None:
            print("   Computing coherence scores...")
            coherence_types = ['c_v', 'c_uci', 'c_npmi', 'u_mass']
            coherences = {}
            for ctype in coherence_types:
                try:
                    cm = CoherenceModel(
                        model=self.lda,
                        texts=self.texts,
                        dictionary=self.dictionary,
                        coherence=ctype
                    )
                    score = cm.get_coherence()
                    coherences[ctype] = float(score)
                    print(f"      {ctype}: {score:.4f}")
                except Exception as e:
                    print(f"      {ctype}: Error - {e}")
                    coherences[ctype] = None
            performance['coherence_scores'] = coherences

        # 2) Perplexity (log perplexity on corpus)
        if self.corpus is not None:
            print("   Computing perplexity...")
            try:
                perplexity = self.lda.log_perplexity(self.corpus)
                performance['perplexity'] = float(perplexity)
                print(f"      Perplexity (log): {perplexity:.4f}")
            except Exception as e:
                print(f"      Perplexity Error: {e}")

        # 3) Topic diversity
        print("   Analyzing topic diversity...")
        diversity_metrics = self._compute_topic_diversity()
        performance['diversity'] = diversity_metrics

        self.results['model_performance'] = performance

    def _compute_topic_diversity(self):
        """Hitung keragaman topik (proporsi kata unik antar topik)"""
        all_topic_words = []
        unique_words = set()

        for topic_id in range(self.lda.num_topics):
            words = [w for w, _ in self.lda.show_topic(topic_id, topn=20)]
            all_topic_words.append(words)
            unique_words.update(words[:10])

        total_top_words = sum(len(words[:10]) for words in all_topic_words)
        overall_diversity = len(unique_words) / total_top_words if total_top_words else 0.0

        topic_diversities = []
        for i, words in enumerate(all_topic_words):
            other_words = set()
            for j, other_words_list in enumerate(all_topic_words):
                if i != j:
                    other_words.update(other_words_list[:10])
            unique_in_topic = len(set(words[:10]) - other_words)
            topic_diversities.append(unique_in_topic / 10.0)

        return {
            'overall_diversity': float(overall_diversity),
            'topic_diversities': [float(d) for d in topic_diversities],
            'avg_topic_diversity': float(np.mean(topic_diversities)),
            'std_topic_diversity': float(np.std(topic_diversities))
        }

    # -------------------------------
    # Topic analysis
    # -------------------------------
    def analyze_topics(self):
        """Analisis mendalam setiap topik"""
        print("\n🔍 Analyzing topics in detail...")

        topics_analysis: List[Dict[str, Any]] = []
        for topic_id in range(self.lda.num_topics):
            topic_words = self.lda.show_topic(topic_id, topn=15)
            words = [w for w, _ in topic_words]
            probs = [float(p) for _, p in topic_words]

            # Optional: auto-interpret (very light heuristic)
            interpretation = self._interpret_topic_automatically(words)

            label = self.topic_labels.get(str(topic_id), f"Topik {topic_id}")
            topics_analysis.append({
                'topic_id': topic_id,
                'label': label,
                'top_words': words,
                'word_probabilities': probs,
                'interpretation': interpretation,
                'word_prob_sum_top10': float(sum(probs[:10])),
                'word_prob_entropy': self._entropy(np.array(probs))
            })

        self.results['topics_analysis'] = topics_analysis

        # Save table: top terms per topic
        rows = []
        for t in topics_analysis:
            for rank, (w, p) in enumerate(zip(t['top_words'], t['word_probabilities']), start=1):
                rows.append({
                    "topic_id": t['topic_id'],
                    "label": t['label'],
                    "rank": rank,
                    "term": w,
                    "weight": p
                })
        pd.DataFrame(rows).to_csv(self.output_dir / "tables" / "top_terms_per_topic.csv", index=False)

    def _interpret_topic_automatically(self, words: List[str]) -> Dict[str, Any]:
        """Interpretasi sederhana (heuristic, tidak menggantikan label manual)"""
        categories = {
            'Politik/Pemerintahan': {'keywords': {'presiden','menteri','pemerintah','partai','pemilu','mk','kpk','politik'}},
            'Ekonomi/Perdagangan': {'keywords': {'ekonomi','usaha','harga','pasar','koperasi','dagang','ekspor'}},
            'Hukum/Peradilan': {'keywords': {'sidang','putusan','perkara','dakwa','jaksa','adil','uu'}},
            'Kriminal/Keamanan': {'keywords': {'polisi','korban','tangkap','duga','aman','tembak'}},
            'Olahraga': {'keywords': {'timnas','laga','gol','menang','liga','pemain','lawan'}},
            'Agama/Sosial': {'keywords': {'jemaah','haji','islam','masyarakat','agama'}},
            'Geopolitik/Internasional': {'keywords': {'israel','iran','rusia','as','gaza','perang'}},
            'Pendidikan': {'keywords': {'sekolah','siswa','guru','didik','ajar'}},
            'Bencana/Kecelakaan': {'keywords': {'gempa','bakar','celaka','kendara'}},
            'Hiburan': {'keywords': {'film','artis','tampil'}},
        }
        sw = set(words[:10])
        best, score = "Topik Umum", 0
        for name, cfg in categories.items():
            overlap = len(sw & cfg['keywords'])
            if overlap > score:
                best, score = name, overlap
        return {"guess": best, "confidence": float(score / 10.0)}

    def _entropy(self, probs: np.ndarray) -> float:
        p = probs / (probs.sum() + 1e-12)
        p = p[p > 0]
        return float(-(p * np.log2(p)).sum()) if len(p) else 0.0

    # -------------------------------
    # Dataset-level topic analysis
    # -------------------------------
    def analyze_dataset_topics(self):
        """Analisis distribusi topik dalam dataset"""
        print("\n📊 Analyzing topic distribution in dataset...")

        if self.corpus is None:
            print("   ⚠️ No corpus available, skipping dataset topic analysis.")
            return

        doc_topics = []
        dominant_topics = []

        for i, bow in enumerate(self.corpus):
            if i and i % 1000 == 0:
                print(f"   ... {i:,}/{len(self.corpus):,} docs")
            dist = self.lda.get_document_topics(bow, minimum_probability=0.0)
            dist = sorted(dist, key=lambda x: -x[1])
            vec = [0.0] * self.lda.num_topics
            for tid, p in dist:
                vec[tid] = p
            doc_topics.append(vec)
            dominant_topics.append(dist[0][0] if dist else None)

        doc_topics = np.array(doc_topics)

        topic_distribution_stats = {
            'dominant_topic_counts': pd.Series(dominant_topics).value_counts().sort_index().to_dict(),
            'avg_topic_probabilities': doc_topics.mean(axis=0).tolist(),
            'std_topic_probabilities': doc_topics.std(axis=0).tolist(),
        }
        self.results['dataset_topic_analysis'] = topic_distribution_stats

        # Save tables for thesis
        # 1) Dominant topic counts
        pd.Series(topic_distribution_stats['dominant_topic_counts']).rename("count") \
            .to_csv(self.output_dir / "tables" / "dominant_topic_counts.csv", header=True)
        # 2) Avg probs per topic
        pd.DataFrame({
            "topic_id": list(range(self.lda.num_topics)),
            "label": [self.topic_labels.get(str(i), f"Topik {i}") for i in range(self.lda.num_topics)],
            "avg_prob": topic_distribution_stats['avg_topic_probabilities'],
            "std_prob": topic_distribution_stats['std_topic_probabilities'],
        }).to_csv(self.output_dir / "tables" / "avg_topic_probabilities.csv", index=False)

        # Optional temporal / by-portal breakdown if df has cols
        if 'released' in self.df.columns:
            print("   ⏱️ Building month-by-topic summary ...")
            # group by month → dominant topic distribution
            grp = self.df.copy()
            grp['month'] = pd.to_datetime(grp['released'], errors='coerce').dt.to_period('M')
            if len(grp['month'].dropna()) == len(dominant_topics):
                grp = grp.dropna(subset=['month']).reset_index(drop=True)
                grp['dom_topic'] = dominant_topics
                dom_month = grp.groupby(['month','dom_topic']).size().unstack(fill_value=0)
                dom_month.to_csv(self.output_dir / "tables" / "dominant_topic_by_month.csv")
        if 'portal' in self.df.columns:
            print("   📰 Building portal-by-topic summary ...")
            if len(self.df) == len(dominant_topics):
                ptab = self.df.copy()
                ptab['dom_topic'] = dominant_topics
                portal_tab = ptab.groupby(['portal','dom_topic']).size().unstack(fill_value=0)
                portal_tab.to_csv(self.output_dir / "tables" / "dominant_topic_by_portal.csv")

    # -------------------------------
    # Visualizations
    # -------------------------------
    def create_visualizations(self):
        print("\n📊 Creating visualizations for thesis...")
        viz_dir = self.output_dir / "visualizations"

        if self.coherence_report is not None:
            self._plot_coherence_comparison(viz_dir)
        self._create_topic_wordclouds(viz_dir)
        self._plot_topic_diversity(viz_dir)
        self._plot_dataset_stats(viz_dir)

        if 'dataset_topic_analysis' in self.results:
            self._plot_topic_distribution(viz_dir)

        self._plot_performance_summary(viz_dir)
        print(f"   ✅ All visualizations saved to {viz_dir}")

    def _plot_coherence_comparison(self, viz_dir: Path):
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        df_sorted = self.coherence_report.sort_values('coherence_cv', ascending=False)
        bars = ax.bar(df_sorted['K'].astype(str), df_sorted['coherence_cv'],
                      color='steelblue', alpha=0.85, edgecolor='navy')
        # highlight best
        ax.bar(df_sorted['K'].astype(str).iloc[0], df_sorted['coherence_cv'].iloc[0],
               color='orange', alpha=0.95, edgecolor='darkorange')
        ax.set_xlabel('Jumlah Topik (K)')
        ax.set_ylabel('Coherence Score (c_v)')
        ax.set_title('Perbandingan Coherence Score (Grid K)')
        ax.grid(axis='y', alpha=0.3)
        for i, (_, row) in enumerate(df_sorted.iterrows()):
            ax.text(i, row['coherence_cv'] + 0.004, f"{row['coherence_cv']:.3f}", ha='center', va='bottom', fontsize=10)
        plt.tight_layout()
        plt.savefig(viz_dir / "coherence_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _create_topic_wordclouds(self, viz_dir: Path):
        print("   Creating topic word clouds...")
        wc_grid_cols = 4
        rows = (self.lda.num_topics + wc_grid_cols - 1) // wc_grid_cols
        fig, axes = plt.subplots(rows, wc_grid_cols, figsize=(20, 5 * rows))
        if rows == 1:
            axes = np.array([axes])
        axes = axes.reshape(rows, wc_grid_cols)

        for topic_id in range(self.lda.num_topics):
            r, c = divmod(topic_id, wc_grid_cols)
            ax = axes[r, c]
            topic_words = dict(self.lda.show_topic(topic_id, topn=30))
            if topic_words:
                wc = WordCloud(width=500, height=300, background_color='white', colormap='viridis',
                               max_words=30).generate_from_frequencies(topic_words)
                ax.imshow(wc, interpolation='bilinear')
                label = self.topic_labels.get(str(topic_id), f"Topik {topic_id}")
                ax.set_title(f"{label} (T{topic_id})", fontsize=12, fontweight='bold')
            else:
                ax.text(0.5, 0.5, 'No words', ha='center', va='center')
                ax.set_title(f"Topik {topic_id}")
            ax.axis('off')

        # remove extra axes
        total = rows * wc_grid_cols
        for k in range(self.lda.num_topics, total):
            r, c = divmod(k, wc_grid_cols)
            axes[r, c].axis('off')

        plt.suptitle('Word Cloud Semua Topik', fontsize=16, y=0.98)
        plt.tight_layout()
        plt.savefig(viz_dir / "all_topic_wordclouds.png", dpi=300, bbox_inches='tight')
        plt.close()

        # individual files
        wc_dir = viz_dir / "wordclouds"
        for topic_id in range(self.lda.num_topics):
            topic_words = dict(self.lda.show_topic(topic_id, topn=40))
            if not topic_words:
                continue
            plt.figure(figsize=(10, 6))
            wc = WordCloud(width=900, height=450, background_color='white', colormap='viridis',
                           max_words=40).generate_from_frequencies(topic_words)
            plt.imshow(wc, interpolation='bilinear')
            label = self.topic_labels.get(str(topic_id), f"Topik {topic_id}")
            plt.title(f"{label} (T{topic_id})", fontsize=16, fontweight='bold')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(wc_dir / f"topic_{topic_id}_wordcloud.png", dpi=300, bbox_inches='tight')
            plt.close()

    def _plot_topic_diversity(self, viz_dir: Path):
        perf = self.results.get('model_performance', {})
        if 'diversity' not in perf:
            return
        diversity = perf['diversity']
        topic_diversities = diversity['topic_diversities']

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        ax1.bar(range(len(topic_diversities)), topic_diversities, color='lightblue', edgecolor='navy', alpha=0.85)
        ax1.axhline(diversity['avg_topic_diversity'], color='red', linestyle='--',
                    label=f"Rata-rata: {diversity['avg_topic_diversity']:.3f}")
        ax1.set_xlabel('Topik ID')
        ax1.set_ylabel('Skor Keragaman (Top-10 unik)')
        ax1.set_title('Keragaman Kata per Topik')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)

        ax2.hist(topic_diversities, bins=10, color='lightgreen', edgecolor='darkgreen', alpha=0.8)
        ax2.axvline(diversity['avg_topic_diversity'], color='red', linestyle='--',
                    label=f"Rata-rata: {diversity['avg_topic_diversity']:.3f}")
        ax2.set_xlabel('Skor Keragaman')
        ax2.set_ylabel('Jumlah Topik')
        ax2.set_title('Distribusi Skor Keragaman Topik')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(viz_dir / "topic_diversity_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_dataset_stats(self, viz_dir: Path):
        if 'dataset_stats' not in self.results:
            return
        stats = self.results['dataset_stats']
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # 1) portal share
        portals = stats.get('portals', {})
        if portals:
            ax1.pie(list(portals.values()), labels=list(portals.keys()), autopct='%1.1f%%', startangle=90)
        ax1.set_title('Distribusi Artikel per Portal')

        # 2) content length
        if 'content_len' in self.df.columns:
            ax2.hist(self.df['content_len'], bins=50, color='skyblue', edgecolor='navy', alpha=0.8)
            if stats.get('avg_content_length') is not None:
                ax2.axvline(stats['avg_content_length'], color='red', linestyle='--',
                            label=f"Avg: {stats['avg_content_length']:.0f}")
            if stats.get('median_content_length') is not None:
                ax2.axvline(stats['median_content_length'], color='orange', linestyle='--',
                            label=f"Median: {stats['median_content_length']:.0f}")
            ax2.set_xlabel('Panjang Artikel (karakter)')
            ax2.set_ylabel('Jumlah Artikel')
            ax2.set_title('Distribusi Panjang Artikel')
            ax2.legend()
            ax2.grid(axis='y', alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'content_len not available', ha='center', va='center')
            ax2.axis('off')

        # 3) Box per portal
        if portals and 'content_len' in self.df.columns:
            data = [self.df[self.df['portal'] == p]['content_len'].values for p in portals.keys()]
            ax3.boxplot(data, labels=list(portals.keys()))
            ax3.set_ylabel('Panjang Artikel (karakter)')
            ax3.set_title('Panjang Artikel per Portal')
            ax3.tick_params(axis='x', rotation=30)
            ax3.grid(axis='y', alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'Portal/content_len not available', ha='center', va='center')
            ax3.axis('off')

        # 4) Summary table
        summary_data = [
            ['Total Artikel', f"{stats.get('total_articles','-')}"],
            ['Jumlah Portal', f"{len(portals)}"],
            ['Avg Panjang', f"{stats.get('avg_content_length','-'):.0f}" if stats.get('avg_content_length') else '-'],
            ['Median Panjang', f"{stats.get('median_content_length','-'):.0f}" if stats.get('median_content_length') else '-'],
            ['Std Panjang', f"{stats.get('content_length_std','-'):.0f}" if stats.get('content_length_std') else '-'],
            ['Periode Data', f"{stats['date_range'].get('start','-')} → {stats['date_range'].get('end','-')}"]
        ]
        ax4.axis('off')
        tb = ax4.table(cellText=summary_data, colLabels=['Metrik', 'Nilai'],
                       cellLoc='center', loc='center')
        tb.auto_set_font_size(False); tb.set_fontsize(10); tb.scale(1.2, 1.5)
        ax4.set_title('Ringkasan Statistik Dataset')

        plt.tight_layout()
        plt.savefig(viz_dir / "dataset_statistics.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_topic_distribution(self, viz_dir: Path):
        analysis = self.results.get('dataset_topic_analysis', {})
        if not analysis:
            return
        dominant_counts = analysis.get('dominant_topic_counts', {})
        avg_probs = analysis.get('avg_topic_probabilities', [])

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # bar of dominant topic counts
        keys = list(dominant_counts.keys())
        vals = [dominant_counts[k] for k in keys]
        ax1.bar(keys, vals, color='lightcoral', edgecolor='darkred', alpha=0.85)
        ax1.set_xlabel('Topik ID')
        ax1.set_ylabel('Jumlah Dokumen')
        ax1.set_title('Distribusi Topik Dominan')

        # avg probs
        ax2.bar(range(len(avg_probs)), avg_probs, color='lightblue', edgecolor='darkblue', alpha=0.85)
        ax2.set_xlabel('Topik ID')
        ax2.set_ylabel('Rata-rata Probabilitas')
        ax2.set_title('Rata-rata Probabilitas Topik (seluruh dataset)')

        plt.tight_layout()
        plt.savefig(viz_dir / "topic_distribution_in_dataset.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_performance_summary(self, viz_dir: Path):
        perf = self.results.get('model_performance', {})
        if not perf:
            return
        fig, ax = plt.subplots(figsize=(8, 4.5))
        lines = []
        coh = perf.get('coherence_scores', {})
        if coh:
            for k, v in coh.items():
                if v is not None:
                    lines.append(f"Coherence {k}: {v:.4f}")
        if 'perplexity' in perf and perf['perplexity'] is not None:
            lines.append(f"Log Perplexity: {perf['perplexity']:.4f}")
        div = perf.get('diversity', {})
        if div:
            lines.append(f"Overall Diversity: {div.get('overall_diversity', 0):.4f}")
            lines.append(f"Avg Topic Diversity: {div.get('avg_topic_diversity', 0):.4f} ± {div.get('std_topic_diversity', 0):.4f}")

        if not lines:
            lines = ["No performance metrics available."]

        ax.axis('off')
        text = "\n".join(lines)
        ax.text(0.01, 0.98, "Ringkasan Performa Model", fontsize=14, fontweight='bold', va='top')
        ax.text(0.03, 0.85, text, fontsize=12, va='top')
        plt.tight_layout()
        plt.savefig(viz_dir / "performance_summary.png", dpi=300, bbox_inches='tight')
        plt.close()

    # -------------------------------
    # Report (Markdown)
    # -------------------------------
    def write_markdown_report(self):
        print("\n📝 Writing markdown report...")
        r = self.output_dir / "reports" / "thesis_report.md"

        def _h(s): return f"## {s}\n"
        def _p(s): return f"{s}\n\n"

        parts = []
        parts.append("# Ringkasan Evaluasi Model LDA\n")
        parts.append(_p(f"Model: **{self.model_dir}** — Topics: **{self.lda.num_topics}** — Vocab: **{len(self.dictionary)}**"))
        ds = self.results.get('dataset_stats', {})
        parts.append(_h("Dataset"))
        parts.append(_p(f"Total artikel: **{ds.get('total_articles','-')}**"))
        if ds.get('date_range'):
            parts.append(_p(f"Periode data: **{ds['date_range'].get('start','-')} → {ds['date_range'].get('end','-')}**"))
        if ds.get('portals'):
            portals = ", ".join([f"{k} ({v})" for k, v in ds['portals'].items()])
            parts.append(_p(f"Portal: {portals}"))

        perf = self.results.get('model_performance', {})
        parts.append(_h("Performa Model"))
        if perf.get('coherence_scores'):
            lines = [f"- {k}: **{v:.4f}**" for k, v in perf['coherence_scores'].items() if v is not None]
            parts.append(_p("\n".join(lines)))
        if perf.get('perplexity') is not None:
            parts.append(_p(f"- Log Perplexity: **{perf['perplexity']:.4f}**"))
        if perf.get('diversity'):
            d = perf['diversity']
            parts.append(_p(f"- Overall Diversity: **{d['overall_diversity']:.4f}**; "
                            f"Avg Topic Diversity: **{d['avg_topic_diversity']:.4f}** ± **{d['std_topic_diversity']:.4f}**"))

        parts.append(_h("Topik & Kata Kunci"))
        parts.append(_p("Lihat tabel: `tables/top_terms_per_topic.csv` dan visual: "
                        "`visualizations/all_topic_wordclouds.png`."))

        if 'dataset_topic_analysis' in self.results:
            parts.append(_h("Distribusi Topik di Dataset"))
            parts.append(_p("Lihat: `visualizations/topic_distribution_in_dataset.png` "
                            "dan tabel `tables/dominant_topic_counts.csv`, `tables/avg_topic_probabilities.csv`."))
        parts.append(_h("Figur Utama untuk Skripsi"))
        parts.append(_p("- `visualizations/coherence_comparison.png`\n"
                        "- `visualizations/all_topic_wordclouds.png`\n"
                        "- `visualizations/topic_diversity_analysis.png`\n"
                        "- `visualizations/dataset_statistics.png`\n"
                        "- `visualizations/topic_distribution_in_dataset.png`\n"
                        "- `visualizations/performance_summary.png`"))

        r.write_text("".join(parts), encoding="utf-8")
        print(f"   ✅ Wrote {r}")

# -------------------------------
# CLI
# -------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, help="Folder berisi lda_best.model, dictionary.dict, phrasers, dll.")
    ap.add_argument("--data_dir", required=True, help="Folder berisi all_berita.csv (+ optional custom_stopwords.txt)")
    ap.add_argument("--output_dir", default="outputs_thesis", help="Folder keluaran visual/tabel/laporan")
    args = ap.parse_args()

    start = time.time()
    eva = ThesisLDAEvaluator(args.model_dir, args.data_dir, args.output_dir)
    eva.load_artifacts()
    eva.load_dataset()
    eva.evaluate_model_performance()
    eva.analyze_topics()
    eva.analyze_dataset_topics()
    eva.create_visualizations()
    eva.write_markdown_report()
    print(f"\n⏱️ Done in {time.time()-start:.1f}s. Outputs → {Path(args.output_dir).resolve()}")

if __name__ == "__main__":
    main()
