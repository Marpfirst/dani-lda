# -*- coding: utf-8 -*-
"""
Preprocessing pipeline untuk korpus berita Indonesia (siap untuk LDA).

Fitur utama:
- Cleaning: hapus URL, emoji, HTML artefak, tanda baca/angka, normalisasi spasi
- Hapus prefix lokasi/portal di awal teks (contoh: "Jakarta, CNN Indonesia - ...")
- Hapus ekspresi waktu WIB (contoh: "pukul 23.30 WIB")
- Tokenisasi sederhana
- Stopword removal (ID umum + stopword domain berita Indonesia)
- Stemming Sastrawi (dengan cache agar cepat) + protected words (opsional)
- N-grams (bigram/trigram) via gensim Phrases

Catatan:
- Gunakan fungsi ini KONSISTEN untuk training & inference.
"""

from typing import List, Iterable, Set, Optional, Tuple
import re
import emoji
from functools import lru_cache
from unidecode import unidecode

from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


# =========================
# Regex & util cleaning
# =========================

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
HTML_ARTIFACT_RE = re.compile(r"(?:&nbsp;|&amp;|&quot;|&lt;|&gt;|&mdash;|&ndash;)", re.I)
# artefak tag sederhana (jaga-jaga kalau ada sisa)
HTML_TAG_TOKEN_RE = re.compile(r"\b(?:nbsp|br|strong|div|span)\b", re.I)

# prefix lokasi/portal di awal teks
LEAD_PORTAL_LOC_RE = re.compile(
    r"^\s*(?:"
    r"jakarta|bandung|surabaya|medan|semarang|yogyakarta|aceh|bali|papua|maluku|kalimantan|sulawesi|nusantara|"
    r"kompas\.com|kompas|cnn indonesia|detikcom|detik|tribunnews\.com|tribunnews|liputan6|tempo|antara|republika"
    r")\s*[,–-]*\s*",
    re.I,
)

# waktu/jam WIB: "pukul 12:30 WIB", "22.15 wib", "10-05 WIB"
WIB_RE = re.compile(r"\b(?:pukul\s*)?\d{1,2}[:.\-]\d{2}\s*wib\b", re.I)

# buang selain huruf/spasi (angka & tanda baca dihapus)
PUNCT_NUM_RE = re.compile(r"[^a-zA-Z\s]")
MULTISPACE_RE = re.compile(r"\s+")


def _normalize_unicode(text: str) -> str:
    text = text.replace("\u200b", " ")  # zero-width space
    text = unidecode(text)
    return text


def base_clean(text: str) -> str:
    """
    Cleaning dasar:
    - unicode normalize, lower
    - hapus URL, emoji, artefak HTML & token HTML
    - hapus prefix lokasi/portal di awal
    - hapus ekspresi WIB (jam)
    - hapus angka/tanda baca, normalisasi spasi
    """
    text = text or ""
    text = _normalize_unicode(text)
    text = URL_RE.sub(" ", text)
    text = emoji.replace_emoji(text, replace=" ")
    text = HTML_ARTIFACT_RE.sub(" ", text)

    # lowercase lebih awal agar regex insensitive lebih efektif
    text = text.lower().strip()

    # prefix lokasi/portal di awal kalimat
    text = LEAD_PORTAL_LOC_RE.sub("", text)

    # hapus ekspresi waktu WIB
    text = WIB_RE.sub(" ", text)

    # buang token html yg lolos
    text = HTML_TAG_TOKEN_RE.sub(" ", text)

    # buang angka/tanda baca
    text = PUNCT_NUM_RE.sub(" ", text)

    # normalisasi spasi
    text = MULTISPACE_RE.sub(" ", text).strip()
    return text


# =========================
# Stopwords
# =========================

def build_stopwords(extra: Iterable[str] = ()) -> Set[str]:
    """
    Gabungkan stopword Bahasa Indonesia + stopword domain berita Indonesia.
    """
    factory = StopWordRemoverFactory()
    base = set(factory.get_stop_words())

    domain = {
        # nama media / portal
        "cnn", "indonesia", "kompas", "kompascom", "detik", "detikcom",
        "tribunnews", "liputan6", "tempo", "antara", "republika", "tribunnewscom", "com",

        # lokasi/fokus geografis umum di awal berita (bukan topik)
        "jakarta", "bandung", "surabaya", "medan", "semarang", "yogyakarta",
        "aceh", "bali", "papua", "maluku", "kalimantan", "sulawesi", "nusantara",

        # waktu & kalender
        "wib", "pukul", "senin", "selasa", "rabu", "kamis", "jumat", "sabtu", "minggu",
        "hari", "bulan", "tahun",
        "januari", "februari", "maret", "april", "mei", "juni", "juli",
        "agustus", "september", "oktober", "november", "desember",

        # gaya jurnalistik (verba kutip & narasi)
        "ujar", "kata", "ungkap", "tutur", "lanjut", "menurut", "menambahkan",
        "menegaskan", "menyebutkan", "ucap", "bilang", "jelas",

        # penanda artikel/hiasan
        "foto", "video", "simak", "selengkapnya", "baca", "klik", "berita",

        # istilah sosial media/meta
        "netizen", "warganet", "trending", "akun", "unggah", "posting", "sosial", "media", "online",

        # artefak tokenisasi/html
        "nbsp", "br", "strong", "div", "span",
    }

    base |= {w.strip().lower() for w in domain}
    base |= {w.strip().lower() for w in extra if w and w.strip()}
    return base


# =========================
# Tokenisasi & stemming
# =========================

def tokenize(text: str) -> List[str]:
    # pisah spasi, buang token panjang < 2 dan non-alfabet
    return [t for t in text.split() if len(t) > 1 and t.isalpha()]


# Stemmer Sastrawi + cache untuk percepat
_stemmer = StemmerFactory().create_stemmer()

@lru_cache(maxsize=200000)
def _stem_token(tok: str) -> str:
    return _stemmer.stem(tok)


def preprocess_document(
    text: str,
    stopwords: Set[str],
    protected_words: Optional[Set[str]] = None,
) -> List[str]:
    """
    Full preprocessing satu dokumen:
      base_clean -> tokenize -> stopword -> stemming (cached) -> stopword lagi
    protected_words: token yang TIDAK akan di-stem (opsional, contoh: {"pemerintah"})
    """
    protected_words = protected_words or set()

    clean = base_clean(text)
    toks = tokenize(clean)

    # stopword awal
    toks = [t for t in toks if t not in stopwords]

    # stemming (dengan proteksi kata tertentu)
    stemmed = []
    for t in toks:
        if t in protected_words:
            stemmed.append(t)
        else:
            stemmed.append(_stem_token(t))

    # stopword ulang (bentuk baru kadang muncul setelah stemming)
    toks = [t for t in stemmed if t not in stopwords]

    return toks


# =========================
# N-grams builder / applier
# =========================

def build_phrasers(
    docs_tokens: List[List[str]],
    min_count: int = 10,
    threshold: float = 10.0,
):
    """
    Bangun bigram & trigram phraser dari dokumen bertoken.
    Naikkan min_count/threshold jika frasa terlalu banyak & tidak informatif.
    """
    from gensim.models.phrases import Phrases, Phraser
    bigram = Phrases(docs_tokens, min_count=min_count, threshold=threshold)
    trigram = Phrases(bigram[docs_tokens], min_count=min_count, threshold=threshold)
    bigram_phraser = Phraser(bigram)
    trigram_phraser = Phraser(trigram)
    return bigram_phraser, trigram_phraser


def apply_ngrams(
    doc_tokens: List[str],
    bigram_phraser=None,
    trigram_phraser=None,
) -> List[str]:
    if bigram_phraser is not None:
        doc_tokens = bigram_phraser[doc_tokens]
    if trigram_phraser is not None:
        doc_tokens = trigram_phraser[doc_tokens]
    return doc_tokens


# =========================
# Contoh penggunaan cepat
# =========================
if __name__ == "__main__":
    text = "JAKARTA, KOMPAS.com – Presiden Jokowi mengumumkan kebijakan baru. Pukul 21.30 WIB."
    stopw = build_stopwords()
    protected = {"pemerintah"}   # contoh proteksi agar tidak menjadi "perintah"
    print(preprocess_document(text, stopw, protected_words=protected))
