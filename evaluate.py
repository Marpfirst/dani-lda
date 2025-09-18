# -*- coding: utf-8 -*-
"""
Pure LDA Evaluator - Sesuai Prinsip Unsupervised Learning
=========================================================
Tool evaluasi model LDA yang murni mengikuti prinsip unsupervised learning
tanpa bias manual labeling atau kategorisasi predetermined.

Usage:
    python pure_evaluate.py --model_dir models --data_dir data --output_dir outputs_pure
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
from sklearn.metrics.pairwise import cosine_similarity

from gensim.models import LdaModel, CoherenceModel
from gensim.corpora import Dictionary

# Optional imports for text rebuilding
try:
    import sys
    here = Path(__file__).resolve().parent
    if (here / "src").exists():
        sys.path.append(str(here / "src"))
    from preprocessing import build_stopwords, preprocess_document, apply_ngrams
    HAVE_PRE = True
except Exception:
    HAVE_PRE = False

# Suppress warnings
warnings.filterwarnings('ignore')

# Matplotlib styling
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16
})


class PureLDAEvaluator:
    """
    Evaluator LDA yang murni mengikuti prinsip unsupervised learning:
    - Tidak ada manual topic labeling
    - Tidak ada kategorisasi predetermined 
    - Evaluasi berdasarkan metrics objektif
    - Interpretasi topik berdasarkan kata-kata yang muncul secara natural
    """
    
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

        print("🎓 Pure LDA Evaluator initialized")
        print(f"   Model dir : {self.model_dir}")
        print(f"   Data dir  : {self.data_dir}")
        print(f"   Output dir: {self.output_dir}")

    def load_artifacts(self):
        """Load model dan artifacts yang diperlukan"""
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

            # Load corpus dan texts jika ada
            corpus_path = self.model_dir / "corpus.pkl"
            if corpus_path.exists():
                with open(corpus_path, "rb") as f:
                    self.corpus = pickle.load(f)

            texts_path = self.model_dir / "processed_texts.pkl"
            if texts_path.exists():
                with open(texts_path, "rb") as f:
                    self.texts = pickle.load(f)

            # Load coherence report untuk perbandingan K
            coherence_path = self.model_dir / "coherence_report.csv"
            self.coherence_report = pd.read_csv(coherence_path) if coherence_path.exists() else None

            print(f"   ✅ Model loaded: {self.lda.num_topics} topics")
            print(f"   ✅ Dictionary: {len(self.dictionary)} words")
            print(f"   ✅ Corpus: {'✓' if self.corpus is not None else '✗'}")
            print(f"   ✅ Texts: {'✓' if self.texts is not None else '✗'}")
            print(f"   ✅ Coherence report: {'✓' if self.coherence_report is not None else '✗'}")

        except Exception as e:
            raise Exception(f"Error loading artifacts: {e}")

    def load_dataset(self):
        """Load dataset asli untuk analisis"""
        print("\n📊 Loading original dataset...")
        csv_path = self.data_dir / "all_berita.csv"
        self.df = pd.read_csv(csv_path)

        # Normalisasi tanggal
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

        # Store dataset statistics
        self.results['dataset_stats'] = {
            'total_articles': int(len(self.df)),
            'portals': (self.df['portal'].value_counts().to_dict()
                        if 'portal' in self.df.columns else {}),
            'avg_content_length': float(self.df.get('content_len', pd.Series(dtype=float)).mean())
                if 'content_len' in self.df.columns else None,
            'median_content_length': float(self.df.get('content_len', pd.Series(dtype=float)).median())
                if 'content_len' in self.df.columns else None,
            'date_range': {
                'start': str(self.df['released'].min()) if 'released' in self.df.columns else None,
                'end': str(self.df['released'].max()) if 'released' in self.df.columns else None
            }
        }

        # Rebuild corpus/texts jika tidak ada
        if (self.corpus is None or self.texts is None) and HAVE_PRE and 'content' in self.df.columns:
            print("   🔧 Rebuilding texts/corpus from raw dataset...")
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

    def evaluate_model_performance(self):
        """Evaluasi performa model dengan metrics objektif"""
        print("\n🧮 Evaluating model performance...")

        performance: Dict[str, Any] = {}

        # 1. Coherence Scores - Metrics utama untuk LDA
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

        # 2. Perplexity
        if self.corpus is not None:
            print("   Computing perplexity...")
            try:
                perplexity = self.lda.log_perplexity(self.corpus)
                performance['perplexity'] = float(perplexity)
                print(f"      Log Perplexity: {perplexity:.4f}")
            except Exception as e:
                print(f"      Perplexity Error: {e}")

        # 3. Topic Quality Metrics
        print("   Computing topic quality metrics...")
        quality_metrics = self._compute_topic_quality_metrics()
        performance.update(quality_metrics)

        self.results['model_performance'] = performance

    def _compute_topic_quality_metrics(self):
        """Compute various topic quality metrics"""
        metrics = {}

        # Topic Diversity (berapa banyak kata unik di top-N kata per topik)
        top_words_per_topic = []
        all_top_words = set()
        
        for topic_id in range(self.lda.num_topics):
            words = [w for w, _ in self.lda.show_topic(topic_id, topn=10)]
            top_words_per_topic.append(set(words))
            all_top_words.update(words)
        
        # Diversity: rasio kata unik total vs total kata (accounting for overlap)
        total_words_if_no_overlap = self.lda.num_topics * 10
        topic_diversity = len(all_top_words) / total_words_if_no_overlap
        metrics['topic_diversity'] = float(topic_diversity)
        
        # Pairwise topic similarity (berdasarkan cosine similarity word distributions)
        topic_similarities = []
        topic_word_matrix = []
        
        # Buat matrix topik-kata
        vocab_size = len(self.dictionary)
        for topic_id in range(self.lda.num_topics):
            topic_dist = np.zeros(vocab_size)
            for word, prob in self.lda.show_topic(topic_id, topn=vocab_size):
                word_id = self.dictionary.token2id.get(word)
                if word_id is not None:
                    topic_dist[word_id] = prob
            topic_word_matrix.append(topic_dist)
        
        topic_word_matrix = np.array(topic_word_matrix)
        
        # Hitung similarity antar topik
        if len(topic_word_matrix) > 1:
            similarity_matrix = cosine_similarity(topic_word_matrix)
            # Ambil upper triangle (tanpa diagonal)
            for i in range(len(similarity_matrix)):
                for j in range(i+1, len(similarity_matrix)):
                    topic_similarities.append(similarity_matrix[i][j])
        
        metrics['avg_topic_similarity'] = float(np.mean(topic_similarities)) if topic_similarities else 0.0
        metrics['std_topic_similarity'] = float(np.std(topic_similarities)) if topic_similarities else 0.0
        
        # Topic Specialization (concentration of probability mass)
        specializations = []
        for topic_id in range(self.lda.num_topics):
            # Hitung entropy dari distribusi kata dalam topik
            topic_words = self.lda.show_topic(topic_id, topn=100)
            probs = np.array([prob for _, prob in topic_words])
            probs = probs / probs.sum()  # normalize
            
            # Entropy: semakin rendah = semakin specialized
            entropy = -np.sum(probs * np.log2(probs + 1e-12))
            specializations.append(entropy)
        
        metrics['avg_topic_specialization'] = float(np.mean(specializations))
        metrics['std_topic_specialization'] = float(np.std(specializations))

        return metrics

    def analyze_topics_unsupervised(self):
        """Analisis topik berdasarkan purely data-driven approach"""
        print("\n🔍 Analyzing topics (unsupervised approach)...")

        topics_analysis: List[Dict[str, Any]] = []
        
        for topic_id in range(self.lda.num_topics):
            topic_words = self.lda.show_topic(topic_id, topn=20)
            words = [w for w, _ in topic_words]
            probs = [float(p) for _, p in topic_words]

            # Analisis statistik kata-kata dalam topik
            word_lengths = [len(w) for w in words]
            
            # Deteksi pola morfologi sederhana
            prefixes = {}
            suffixes = {}
            for word in words[:10]:  # top 10 words
                if len(word) > 3:
                    prefix = word[:3]
                    suffix = word[-3:]
                    prefixes[prefix] = prefixes.get(prefix, 0) + 1
                    suffixes[suffix] = suffixes.get(suffix, 0) + 1
            
            # Most common patterns
            common_prefix = max(prefixes.items(), key=lambda x: x[1])[0] if prefixes else None
            common_suffix = max(suffixes.items(), key=lambda x: x[1])[0] if suffixes else None

            topics_analysis.append({
                'topic_id': topic_id,
                'top_words': words,
                'word_probabilities': probs,
                'word_prob_concentration': float(sum(probs[:5]) / sum(probs)),  # konsentrasi prob di top-5
                'avg_word_length': float(np.mean(word_lengths)),
                'word_length_std': float(np.std(word_lengths)),
                'most_common_prefix': common_prefix,
                'most_common_suffix': common_suffix,
                'topic_entropy': self._entropy(np.array(probs))
            })

        self.results['topics_analysis'] = topics_analysis

        # Save topic terms table
        rows = []
        for t in topics_analysis:
            for rank, (word, prob) in enumerate(zip(t['top_words'], t['word_probabilities']), 1):
                rows.append({
                    "topic_id": t['topic_id'],
                    "rank": rank,
                    "term": word,
                    "probability": prob
                })
        
        df_terms = pd.DataFrame(rows)
        df_terms.to_csv(self.output_dir / "tables" / "topic_terms.csv", index=False)

        # Save topic statistics
        topic_stats = []
        for t in topics_analysis:
            topic_stats.append({
                'topic_id': t['topic_id'],
                'top_3_words': ' + '.join(t['top_words'][:3]),
                'prob_concentration_top5': t['word_prob_concentration'],
                'avg_word_length': t['avg_word_length'],
                'topic_entropy': t['topic_entropy'],
                'common_prefix': t['most_common_prefix'],
                'common_suffix': t['most_common_suffix']
            })
        
        pd.DataFrame(topic_stats).to_csv(
            self.output_dir / "tables" / "topic_statistics.csv", index=False
        )

    def analyze_document_topic_distributions(self):
        """Analisis distribusi topik dalam dokumen"""
        print("\n📊 Analyzing document-topic distributions...")

        if self.corpus is None:
            print("   ⚠️ No corpus available, skipping document analysis.")
            return

        # Compute document-topic distributions
        doc_topics_list = []
        dominant_topics = []
        topic_concentrations = []  # untuk mengukur seberapa "focused" dokumen pada topik tertentu

        print(f"   Processing {len(self.corpus):,} documents...")
        for i, bow in enumerate(self.corpus):
            if i and i % 1000 == 0:
                print(f"   ... {i:,}/{len(self.corpus):,}")
            
            # Get topic distribution for document
            doc_topics = self.lda.get_document_topics(bow, minimum_probability=0.0)
            doc_topics = sorted(doc_topics, key=lambda x: -x[1])
            
            # Convert to array
            topic_dist = np.zeros(self.lda.num_topics)
            for topic_id, prob in doc_topics:
                topic_dist[topic_id] = prob
            
            doc_topics_list.append(topic_dist)
            dominant_topics.append(doc_topics[0][0] if doc_topics else None)
            
            # Topic concentration (entropy-based measure)
            non_zero_probs = topic_dist[topic_dist > 0]
            if len(non_zero_probs) > 0:
                concentration = -np.sum(non_zero_probs * np.log2(non_zero_probs))
                topic_concentrations.append(concentration)
            else:
                topic_concentrations.append(0)

        doc_topics_matrix = np.array(doc_topics_list)

        # Compute statistics
        analysis = {
            'dominant_topic_distribution': pd.Series(dominant_topics).value_counts().sort_index().to_dict(),
            'avg_topic_probabilities': doc_topics_matrix.mean(axis=0).tolist(),
            'std_topic_probabilities': doc_topics_matrix.std(axis=0).tolist(),
            'avg_topic_concentration': float(np.mean(topic_concentrations)),
            'std_topic_concentration': float(np.std(topic_concentrations)),
        }

        self.results['document_topic_analysis'] = analysis

        # Save tables
        # 1. Dominant topic counts
        dom_topic_df = pd.DataFrame([
            {'topic_id': k, 'document_count': v, 'percentage': v/len(dominant_topics)*100}
            for k, v in analysis['dominant_topic_distribution'].items()
        ])
        dom_topic_df.to_csv(self.output_dir / "tables" / "dominant_topics.csv", index=False)

        # 2. Average topic probabilities
        avg_prob_df = pd.DataFrame({
            'topic_id': range(self.lda.num_topics),
            'avg_probability': analysis['avg_topic_probabilities'],
            'std_probability': analysis['std_topic_probabilities']
        })
        avg_prob_df.to_csv(self.output_dir / "tables" / "avg_topic_probabilities.csv", index=False)

        # 3. Temporal analysis if possible
        if 'released' in self.df.columns and len(self.df) == len(dominant_topics):
            print("   Creating temporal topic analysis...")
            temporal_df = self.df.copy()
            temporal_df['dominant_topic'] = dominant_topics
            temporal_df['month'] = pd.to_datetime(temporal_df['released'], errors='coerce').dt.to_period('M')
            
            if temporal_df['month'].notna().any():
                temporal_pivot = temporal_df.groupby(['month', 'dominant_topic']).size().unstack(fill_value=0)
                temporal_pivot.to_csv(self.output_dir / "tables" / "temporal_topic_trends.csv")

    def create_visualizations(self):
        """Create visualizations untuk analisis"""
        print("\n📊 Creating visualizations...")
        viz_dir = self.output_dir / "visualizations"

        # 1. Coherence comparison jika ada
        if self.coherence_report is not None:
            self._plot_coherence_optimization(viz_dir)

        # 2. Topic word clouds
        self._create_topic_wordclouds(viz_dir)

        # 3. Topic quality metrics
        self._plot_topic_quality_metrics(viz_dir)

        # 4. Document-topic analysis
        if 'document_topic_analysis' in self.results:
            self._plot_document_topic_analysis(viz_dir)

        # 5. Dataset overview
        self._plot_dataset_overview(viz_dir)

        print(f"   ✅ All visualizations saved to {viz_dir}")

    def _plot_coherence_optimization(self, viz_dir: Path):
        """Plot coherence scores untuk berbagai K"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        df = self.coherence_report.sort_values('K')
        
        # Plot coherence score
        ax.plot(df['K'], df['coherence_cv'], 'o-', linewidth=2, markersize=8, 
                color='steelblue', label='Coherence Score (C_V)')
        
        # Highlight optimal K
        best_k = df.loc[df['coherence_cv'].idxmax(), 'K']
        best_score = df['coherence_cv'].max()
        ax.scatter([best_k], [best_score], color='red', s=100, zorder=5, 
                  label=f'Optimal K={best_k}')
        
        ax.set_xlabel('Number of Topics (K)')
        ax.set_ylabel('Coherence Score (C_V)')
        ax.set_title('Topic Number Optimization via Coherence Score')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Annotate values
        for _, row in df.iterrows():
            ax.annotate(f'{row["coherence_cv"]:.3f}', 
                       (row['K'], row['coherence_cv']), 
                       textcoords="offset points", xytext=(0,10), ha='center')
        
        plt.tight_layout()
        plt.savefig(viz_dir / "coherence_optimization.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _create_topic_wordclouds(self, viz_dir: Path):
        """Create word clouds untuk setiap topik"""
        print("   Creating topic word clouds...")
        
        # Grid layout
        cols = 4
        rows = (self.lda.num_topics + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
        
        if rows == 1:
            axes = np.array([axes])
        axes = axes.reshape(rows, cols)

        for topic_id in range(self.lda.num_topics):
            r, c = divmod(topic_id, cols)
            ax = axes[r, c]
            
            # Get topic words
            topic_words = dict(self.lda.show_topic(topic_id, topn=30))
            
            if topic_words:
                # Create word cloud
                wc = WordCloud(
                    width=400, height=300, 
                    background_color='white',
                    colormap='viridis',
                    max_words=30,
                    relative_scaling=0.5,
                    min_font_size=8
                ).generate_from_frequencies(topic_words)
                
                ax.imshow(wc, interpolation='bilinear')
                
                # Title dengan top 3 kata
                top_words = list(topic_words.keys())[:3]
                title = f"Topic {topic_id}: {' + '.join(top_words)}"
                ax.set_title(title, fontsize=11, pad=10)
            else:
                ax.text(0.5, 0.5, 'No words', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f"Topic {topic_id}")
            
            ax.axis('off')

        # Remove extra subplots
        for k in range(self.lda.num_topics, rows * cols):
            r, c = divmod(k, cols)
            axes[r, c].axis('off')

        plt.suptitle(f'Topic Word Clouds (K={self.lda.num_topics})', fontsize=16, y=0.98)
        plt.tight_layout()
        plt.savefig(viz_dir / "topic_wordclouds_grid.png", dpi=300, bbox_inches='tight')
        plt.close()

        # Individual word clouds
        wc_dir = viz_dir / "wordclouds"
        for topic_id in range(self.lda.num_topics):
            topic_words = dict(self.lda.show_topic(topic_id, topn=50))
            
            if topic_words:
                plt.figure(figsize=(12, 8))
                
                wc = WordCloud(
                    width=800, height=600,
                    background_color='white',
                    colormap='viridis',
                    max_words=50,
                    relative_scaling=0.5,
                    min_font_size=10
                ).generate_from_frequencies(topic_words)
                
                plt.imshow(wc, interpolation='bilinear')
                
                top_words = list(topic_words.keys())[:5]
                title = f"Topic {topic_id}: {' + '.join(top_words)}"
                plt.title(title, fontsize=16, pad=20)
                plt.axis('off')
                
                plt.tight_layout()
                plt.savefig(wc_dir / f"topic_{topic_id:02d}_wordcloud.png", 
                           dpi=300, bbox_inches='tight')
                plt.close()

    def _plot_topic_quality_metrics(self, viz_dir: Path):
        """Plot topic quality metrics"""
        perf = self.results.get('model_performance', {})
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Coherence scores
        coherence_scores = perf.get('coherence_scores', {})
        if coherence_scores:
            metrics = []
            values = []
            for metric, value in coherence_scores.items():
                if value is not None:
                    metrics.append(metric.upper())
                    values.append(value)
            
            bars = ax1.bar(metrics, values, color='skyblue', edgecolor='navy', alpha=0.8)
            ax1.set_title('Coherence Scores')
            ax1.set_ylabel('Score')
            ax1.grid(axis='y', alpha=0.3)
            
            # Annotate values
            for bar, val in zip(bars, values):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                        f'{val:.3f}', ha='center', va='bottom')

        # 2. Topic diversity
        diversity = perf.get('topic_diversity', 0)
        ax2.bar(['Topic Diversity'], [diversity], color='lightgreen', edgecolor='darkgreen', alpha=0.8)
        ax2.set_title('Topic Diversity Score')
        ax2.set_ylabel('Diversity (0-1)')
        ax2.set_ylim(0, 1)
        ax2.text(0, diversity + 0.02, f'{diversity:.3f}', ha='center', va='bottom')

        # 3. Topic similarity distribution
        avg_sim = perf.get('avg_topic_similarity', 0)
        std_sim = perf.get('std_topic_similarity', 0)
        ax3.bar(['Avg Topic Similarity'], [avg_sim], yerr=[std_sim], 
                color='salmon', edgecolor='darkred', alpha=0.8, capsize=5)
        ax3.set_title('Inter-Topic Similarity')
        ax3.set_ylabel('Cosine Similarity')
        ax3.text(0, avg_sim + std_sim + 0.01, f'{avg_sim:.3f}±{std_sim:.3f}', 
                ha='center', va='bottom')

        # 4. Topic specialization
        avg_spec = perf.get('avg_topic_specialization', 0)
        std_spec = perf.get('std_topic_specialization', 0)
        ax4.bar(['Avg Topic Specialization'], [avg_spec], yerr=[std_spec],
                color='gold', edgecolor='orange', alpha=0.8, capsize=5)
        ax4.set_title('Topic Specialization (Entropy)')
        ax4.set_ylabel('Entropy')
        ax4.text(0, avg_spec + std_spec + 0.1, f'{avg_spec:.2f}±{std_spec:.2f}', 
                ha='center', va='bottom')

        plt.suptitle('Model Quality Metrics', fontsize=16)
        plt.tight_layout()
        plt.savefig(viz_dir / "topic_quality_metrics.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_document_topic_analysis(self, viz_dir: Path):
        """Plot analisis dokumen-topik"""
        analysis = self.results['document_topic_analysis']
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Dominant topic distribution
        dom_topics = analysis['dominant_topic_distribution']
        topics = list(dom_topics.keys())
        counts = list(dom_topics.values())
        
        bars = ax1.bar(topics, counts, color='lightcoral', edgecolor='darkred', alpha=0.8)
        ax1.set_xlabel('Topic ID')
        ax1.set_ylabel('Number of Documents')
        ax1.set_title('Dominant Topic Distribution Across Documents')
        ax1.grid(axis='y', alpha=0.3)
        
        # 2. Average topic probabilities
        avg_probs = analysis['avg_topic_probabilities']
        std_probs = analysis['std_topic_probabilities']
        
        ax2.bar(range(len(avg_probs)), avg_probs, yerr=std_probs,
               color='lightblue', edgecolor='darkblue', alpha=0.8, capsize=3)
        ax2.set_xlabel('Topic ID')
        ax2.set_ylabel('Average Probability')
        ax2.set_title('Average Topic Probabilities Across All Documents')
        ax2.grid(axis='y', alpha=0.3)
        
        # 3. Topic concentration histogram
        avg_concentration = analysis['avg_topic_concentration']
        std_concentration = analysis['std_topic_concentration']
        
        # Create a simple representation since we don't have individual concentrations
        ax3.bar(['Document Focus'], [avg_concentration], yerr=[std_concentration],
               color='gold', edgecolor='orange', alpha=0.8, capsize=5)
        ax3.set_ylabel('Average Entropy (Topic Concentration)')
        ax3.set_title('Document Topic Concentration\n(Lower = More Focused)')
        ax3.text(0, avg_concentration + std_concentration + 0.1, 
                f'{avg_concentration:.2f}±{std_concentration:.2f}', 
                ha='center', va='bottom')
        
        # 4. Topic coverage (how many topics are "active" on average)
        # Calculate based on avg probabilities > threshold
        active_topics_001 = sum(1 for p in avg_probs if p > 0.01)
        active_topics_005 = sum(1 for p in avg_probs if p > 0.05)
        active_topics_010 = sum(1 for p in avg_probs if p > 0.10)
        
        thresholds = ['> 1%', '> 5%', '> 10%']
        active_counts = [active_topics_001, active_topics_005, active_topics_010]
        
        bars = ax4.bar(thresholds, active_counts, 
                      color=['lightgreen', 'yellow', 'orange'], 
                      edgecolor='black', alpha=0.8)
        ax4.set_ylabel('Number of Active Topics')
        ax4.set_title('Topic Activity Levels\n(Based on Avg Probability Threshold)')
        ax4.set_ylim(0, self.lda.num_topics + 1)
        
        # Annotate bars
        for bar, count in zip(bars, active_counts):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(count), ha='center', va='bottom')

        plt.suptitle('Document-Topic Analysis', fontsize=16)
        plt.tight_layout()
        plt.savefig(viz_dir / "document_topic_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_dataset_overview(self, viz_dir: Path):
        """Plot overview dataset"""
        stats = self.results.get('dataset_stats', {})
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Portal distribution
        portals = stats.get('portals', {})
        if portals:
            # Get top 10 portals
            portal_items = sorted(portals.items(), key=lambda x: x[1], reverse=True)[:10]
            names, counts = zip(*portal_items)
            
            ax1.barh(range(len(names)), counts, color='steelblue', alpha=0.8)
            ax1.set_yticks(range(len(names)))
            ax1.set_yticklabels(names)
            ax1.set_xlabel('Number of Articles')
            ax1.set_title('Top 10 News Portals by Article Count')
            ax1.grid(axis='x', alpha=0.3)
        else:
            ax1.text(0.5, 0.5, 'Portal data not available', 
                    ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Portal Distribution')
        
        # 2. Content length distribution
        if 'content_len' in self.df.columns:
            content_lengths = self.df['content_len'].dropna()
            ax2.hist(content_lengths, bins=50, color='skyblue', 
                    edgecolor='navy', alpha=0.8)
            
            # Add statistics lines
            mean_len = content_lengths.mean()
            median_len = content_lengths.median()
            
            ax2.axvline(mean_len, color='red', linestyle='--', 
                       label=f'Mean: {mean_len:.0f}')
            ax2.axvline(median_len, color='orange', linestyle='--', 
                       label=f'Median: {median_len:.0f}')
            
            ax2.set_xlabel('Article Length (characters)')
            ax2.set_ylabel('Number of Articles')
            ax2.set_title('Article Length Distribution')
            ax2.legend()
            ax2.grid(axis='y', alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'Content length data not available', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Article Length Distribution')
        
        # 3. Temporal distribution (if available)
        if 'released' in self.df.columns:
            try:
                # Monthly article counts
                monthly_counts = self.df.groupby(
                    pd.to_datetime(self.df['released']).dt.to_period('M')
                ).size()
                
                ax3.plot(monthly_counts.index.astype(str), monthly_counts.values, 
                        'o-', linewidth=2, markersize=4, color='green')
                ax3.set_xlabel('Month')
                ax3.set_ylabel('Number of Articles')
                ax3.set_title('Temporal Distribution of Articles')
                ax3.tick_params(axis='x', rotation=45)
                ax3.grid(True, alpha=0.3)
                
                # Show only every nth tick to avoid crowding
                n_ticks = min(10, len(monthly_counts))
                tick_positions = np.linspace(0, len(monthly_counts)-1, n_ticks, dtype=int)
                ax3.set_xticks([monthly_counts.index[i] for i in tick_positions])
                ax3.set_xticklabels([str(monthly_counts.index[i]) for i in tick_positions])
                
            except Exception:
                ax3.text(0.5, 0.5, 'Could not parse temporal data', 
                        ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title('Temporal Distribution')
        else:
            ax3.text(0.5, 0.5, 'Temporal data not available', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Temporal Distribution')
        
        # 4. Dataset summary statistics
        ax4.axis('off')
        
        summary_data = [
            ['Total Articles', f"{stats.get('total_articles', 'N/A'):,}"],
            ['Number of Portals', f"{len(portals)}"],
            ['Date Range', f"{stats.get('date_range', {}).get('start', 'N/A')} to\n{stats.get('date_range', {}).get('end', 'N/A')}"],
            ['Avg Content Length', f"{stats.get('avg_content_length', 0):.0f}" if stats.get('avg_content_length') else 'N/A'],
            ['Model Topics', f"{self.lda.num_topics}"],
            ['Vocabulary Size', f"{len(self.dictionary):,}"]
        ]
        
        table = ax4.table(cellText=summary_data, 
                         colLabels=['Metric', 'Value'],
                         cellLoc='left', loc='center',
                         colWidths=[0.4, 0.6])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 2)
        
        # Style the table
        for i in range(len(summary_data) + 1):
            for j in range(2):
                cell = table[(i, j)]
                if i == 0:  # Header
                    cell.set_facecolor('#4CAF50')
                    cell.set_text_props(weight='bold', color='white')
                else:
                    cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
        
        ax4.set_title('Dataset Summary', fontsize=14, fontweight='bold', pad=20)

        plt.suptitle('Dataset Overview', fontsize=16)
        plt.tight_layout()
        plt.savefig(viz_dir / "dataset_overview.png", dpi=300, bbox_inches='tight')
        plt.close()

    def generate_markdown_report(self):
        """Generate comprehensive markdown report"""
        print("\n📝 Writing comprehensive markdown report...")
        
        report_path = self.output_dir / "reports" / "pure_lda_evaluation_report.md"
        
        # Build report content
        content = []
        content.append("# Pure LDA Model Evaluation Report\n")
        content.append("*Generated using unsupervised evaluation approach*\n\n")
        
        # Model Overview
        content.append("## Model Overview\n")
        content.append(f"- **Number of Topics (K)**: {self.lda.num_topics}\n")
        content.append(f"- **Vocabulary Size**: {len(self.dictionary):,} unique terms\n")
        content.append(f"- **Model Type**: Latent Dirichlet Allocation (LDA)\n")
        content.append(f"- **Evaluation Approach**: Pure unsupervised (no manual labeling)\n\n")
        
        # Dataset Statistics
        stats = self.results.get('dataset_stats', {})
        content.append("## Dataset Statistics\n")
        content.append(f"- **Total Articles**: {stats.get('total_articles', 'N/A'):,}\n")
        content.append(f"- **Number of Portals**: {len(stats.get('portals', {}))}\n")
        
        if stats.get('date_range'):
            date_range = stats['date_range']
            content.append(f"- **Date Range**: {date_range.get('start', 'N/A')} to {date_range.get('end', 'N/A')}\n")
        
        if stats.get('avg_content_length'):
            content.append(f"- **Average Article Length**: {stats['avg_content_length']:.0f} characters\n")
            content.append(f"- **Median Article Length**: {stats.get('median_content_length', 0):.0f} characters\n")
        
        content.append("\n")
        
        # Model Performance
        perf = self.results.get('model_performance', {})
        content.append("## Model Performance Metrics\n")
        
        # Coherence scores
        coherence = perf.get('coherence_scores', {})
        if coherence:
            content.append("### Coherence Scores\n")
            for metric, score in coherence.items():
                if score is not None:
                    content.append(f"- **{metric.upper()}**: {score:.4f}\n")
            content.append("\n")
        
        # Perplexity
        if 'perplexity' in perf:
            content.append(f"### Perplexity\n")
            content.append(f"- **Log Perplexity**: {perf['perplexity']:.4f}\n")
            content.append("  *(Lower values indicate better model fit)*\n\n")
        
        # Topic Quality Metrics
        content.append("### Topic Quality Metrics\n")
        if 'topic_diversity' in perf:
            content.append(f"- **Topic Diversity**: {perf['topic_diversity']:.4f}\n")
            content.append("  *(Higher values indicate more diverse topics)*\n")
        
        if 'avg_topic_similarity' in perf:
            content.append(f"- **Average Inter-Topic Similarity**: {perf['avg_topic_similarity']:.4f} ± {perf.get('std_topic_similarity', 0):.4f}\n")
            content.append("  *(Lower values indicate more distinct topics)*\n")
        
        if 'avg_topic_specialization' in perf:
            content.append(f"- **Average Topic Specialization**: {perf['avg_topic_specialization']:.4f} ± {perf.get('std_topic_specialization', 0):.4f}\n")
            content.append("  *(Lower entropy indicates more specialized topics)*\n")
        
        content.append("\n")
        
        # Topic Analysis
        topics = self.results.get('topics_analysis', [])
        if topics:
            content.append("## Discovered Topics\n")
            content.append("*Topics are presented with their most characteristic terms as discovered by the model*\n\n")
            
            for topic in topics:
                topic_id = topic['topic_id']
                top_words = topic['top_words'][:8]  # Top 8 words
                
                content.append(f"### Topic {topic_id}\n")
                content.append(f"**Key Terms**: {' • '.join(top_words)}\n")
                content.append(f"- Probability Concentration (top-5): {topic['word_prob_concentration']:.3f}\n")
                content.append(f"- Topic Entropy: {topic['topic_entropy']:.3f}\n")
                content.append(f"- Average Word Length: {topic['avg_word_length']:.1f} characters\n")
                
                if topic.get('most_common_prefix') or topic.get('most_common_suffix'):
                    content.append("- Morphological patterns: ")
                    patterns = []
                    if topic.get('most_common_prefix'):
                        patterns.append(f"prefix '{topic['most_common_prefix']}'")
                    if topic.get('most_common_suffix'):
                        patterns.append(f"suffix '{topic['most_common_suffix']}'")
                    content.append(", ".join(patterns) + "\n")
                
                content.append("\n")
        
        # Document-Topic Analysis
        doc_analysis = self.results.get('document_topic_analysis', {})
        if doc_analysis:
            content.append("## Document-Topic Distribution Analysis\n")
            
            dom_topics = doc_analysis['dominant_topic_distribution']
            total_docs = sum(dom_topics.values())
            
            content.append("### Topic Prevalence\n")
            sorted_topics = sorted(dom_topics.items(), key=lambda x: x[1], reverse=True)
            
            for topic_id, count in sorted_topics[:5]:  # Top 5 most prevalent
                percentage = (count / total_docs) * 100
                content.append(f"- **Topic {topic_id}**: {count:,} documents ({percentage:.1f}%)\n")
            
            content.append(f"\n### Document Focus\n")
            content.append(f"- **Average Topic Concentration**: {doc_analysis['avg_topic_concentration']:.3f} ± {doc_analysis['std_topic_concentration']:.3f}\n")
            content.append("  *(Lower entropy indicates documents are more focused on specific topics)*\n\n")
        
        # Files Generated
        content.append("## Generated Files\n")
        content.append("### Visualizations\n")
        viz_files = [
            ("coherence_optimization.png", "Topic number optimization via coherence scores"),
            ("topic_wordclouds_grid.png", "Word clouds for all topics in grid layout"),
            ("topic_quality_metrics.png", "Comprehensive topic quality metrics"),
            ("document_topic_analysis.png", "Document-topic distribution analysis"),
            ("dataset_overview.png", "Dataset statistics and overview")
        ]
        
        for filename, description in viz_files:
            content.append(f"- `visualizations/{filename}`: {description}\n")
        
        content.append("\n### Data Tables\n")
        table_files = [
            ("topic_terms.csv", "Top terms for each topic with probabilities"),
            ("topic_statistics.csv", "Statistical analysis of topic characteristics"),
            ("dominant_topics.csv", "Distribution of dominant topics across documents"),
            ("avg_topic_probabilities.csv", "Average topic probabilities across all documents")
        ]
        
        for filename, description in table_files:
            content.append(f"- `tables/{filename}`: {description}\n")
        
        content.append("\n### Individual Word Clouds\n")
        content.append(f"- `visualizations/wordclouds/`: Individual high-resolution word clouds for each topic\n")
        
        # Methodology Notes
        content.append("\n## Methodology Notes\n")
        content.append("This evaluation follows pure unsupervised learning principles:\n\n")
        content.append("1. **No Manual Labeling**: Topics are identified purely by their word distributions\n")
        content.append("2. **Objective Metrics**: Evaluation relies on established metrics (coherence, perplexity, diversity)\n")
        content.append("3. **Data-Driven Analysis**: Topic characteristics are derived from statistical analysis\n")
        content.append("4. **Reproducible Results**: All analyses can be reproduced using the same model and data\n\n")
        
        content.append("### Interpretation Guidelines\n")
        content.append("- Topic quality should be assessed primarily through coherence scores\n")
        content.append("- High topic diversity indicates the model discovered distinct themes\n")
        content.append("- Low inter-topic similarity suggests topics are well-separated\n")
        content.append("- Document topic concentration indicates how focused articles are on specific themes\n")
        
        # Write report
        report_content = "".join(content)
        report_path.write_text(report_content, encoding='utf-8')
        
        print(f"   ✅ Report written to {report_path}")
        return report_path

    def _entropy(self, probs: np.ndarray) -> float:
        """Calculate entropy of probability distribution"""
        p = probs / (probs.sum() + 1e-12)
        p = p[p > 0]
        return float(-(p * np.log2(p)).sum()) if len(p) > 0 else 0.0

    def run_complete_evaluation(self):
        """Run complete evaluation pipeline"""
        print("\n🚀 Running complete LDA evaluation pipeline...")
        
        start_time = time.time()
        
        # Load everything
        self.load_artifacts()
        self.load_dataset()
        
        # Run evaluations
        self.evaluate_model_performance()
        self.analyze_topics_unsupervised()
        self.analyze_document_topic_distributions()
        
        # Generate outputs
        self.create_visualizations()
        report_path = self.generate_markdown_report()
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n✅ Evaluation completed in {duration:.1f} seconds")
        print(f"📁 All outputs saved to: {self.output_dir.resolve()}")
        print(f"📊 Main report: {report_path}")
        
        return self.results


def main():
    parser = argparse.ArgumentParser(description="Pure LDA Evaluator - Unsupervised Approach")
    parser.add_argument("--model_dir", required=True, 
                       help="Directory containing LDA model files")
    parser.add_argument("--data_dir", required=True,
                       help="Directory containing dataset (all_berita.csv)")
    parser.add_argument("--output_dir", default="outputs_pure_lda",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    # Initialize and run evaluator
    evaluator = PureLDAEvaluator(args.model_dir, args.data_dir, args.output_dir)
    results = evaluator.run_complete_evaluation()
    
    print("\n" + "="*60)
    print("🎯 PURE LDA EVALUATION SUMMARY")
    print("="*60)
    
    # Print key metrics
    perf = results.get('model_performance', {})
    if perf.get('coherence_scores', {}).get('c_v'):
        print(f"📊 Coherence (C_V): {perf['coherence_scores']['c_v']:.4f}")
    if perf.get('perplexity'):
        print(f"📈 Log Perplexity: {perf['perplexity']:.4f}")
    if perf.get('topic_diversity'):
        print(f"🔀 Topic Diversity: {perf['topic_diversity']:.4f}")
    
    dataset_stats = results.get('dataset_stats', {})
    if dataset_stats.get('total_articles'):
        print(f"📰 Total Articles: {dataset_stats['total_articles']:,}")
    
    print(f"🎯 Topics Discovered: {evaluator.lda.num_topics}")
    print(f"📁 Results: {evaluator.output_dir.resolve()}")


if __name__ == "__main__":
    main()