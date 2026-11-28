"""Pembacaan manifest, split antar penutur, dan atribut turunan untuk analisis."""

import csv
import re
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data" / "manifest_normalized.csv"
SPLITS = ROOT / "data" / "splits"
SPEAKERS = ROOT / "data" / "speakers.csv"
FOLDS = [1, 2, 3, 4, 5]

TAG_RE = re.compile(r"\[[^\]]*\]")


def read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_manifest(path: Path = MANIFEST) -> List[Dict[str, str]]:
    """Baris manifest yang usable (punya teks dan waktu valid)."""
    return [r for r in read_csv(path) if r.get("usable", "yes") == "yes"]


def split_map(fold: int) -> Dict[str, str]:
    """utt_id -> train | val | test pada satu fold."""
    return {r["utt_id"]: r["split"] for r in read_csv(SPLITS / f"fold{fold}.csv")}


def test_fold_by_speaker() -> Dict[str, int]:
    """Setiap penutur menjadi test tepat satu kali; kembalikan nomor fold-nya."""
    out = {}
    for r in read_csv(SPLITS / "speaker_folds.csv"):
        folds = [k for k in FOLDS if r[f"fold{k}"] == "test"]
        if len(folds) != 1:
            raise ValueError(f"{r['speaker']} harus menjadi test tepat sekali, dapat {folds}")
        out[r["speaker"]] = folds[0]
    return out


def speaker_table() -> Dict[str, Dict[str, str]]:
    return {r["speaker"]: r for r in read_csv(SPEAKERS)}


def dialect_family(label: str) -> str:
    """Kelompokkan label dialek tingkat penutur (konvensi dua korpus berbeda)."""
    low = (label or "").lower()
    if "campuran" in low:
        return "Campuran"
    if "timuran" in low:
        return "Timuran"
    if any(k in low for k in ("solo", "jogja", "mataraman")):
        return "Solo-Jogja (Mataraman)"
    if "semarang" in low:
        return "Semarangan"
    return "Tidak dirinci"


EVENT_PATTERNS = {
    "noise latar/musik/hewan": ("musik", "background_noise", "hewan"),
    "napas/batuk/tawa": ("napas", "batuk", "tawa", "cough", "laughter"),
    "disfluensi": ("pengulangan", "filler", "filled_pause"),
    "tidak jelas/terpotong": ("unclear", "truncated", "kurang yakin", "overlap"),
    "catatan masalah ASR/hening": ("halusinasi", "asr mengulang", "hening"),
    "membaca pertanyaan panduan": ("membacakan pertanyaan",),
}


def event_categories(event: str) -> List[str]:
    """Kategori kejadian satu segmen (bisa lebih dari satu)."""
    low = (event or "").strip().lower()
    if low in ("", "-", "none"):
        return ["tanpa label kejadian"]
    found = [name for name, keys in EVENT_PATTERNS.items() if any(k in low for k in keys)]
    return found or ["kejadian lain"]


def duration_bucket(seconds: float) -> str:
    if seconds < 1:
        return "a. < 1 s"
    if seconds < 3:
        return "b. 1-3 s"
    if seconds < 10:
        return "c. 3-10 s"
    return "d. >= 10 s"


def clean_exclusion_reasons(row: Dict[str, str]) -> List[str]:
    """Alasan sebuah segmen dikeluarkan dari subset 'bersih' (evaluasi pendamping).

    Subset ini hanya dipakai sebagai angka pembanding; hasil utama tetap memakai semua segmen.
    """
    reasons = []
    event = (row.get("event") or "").lower()
    if "membacakan pertanyaan" in event:
        reasons.append("pertanyaan_panduan")
    if "hening digital" in event or "halusinasi" in event:
        reasons.append("hening_atau_halusinasi")
    if row.get("language_label") in ("Indonesia", "Inggris"):
        reasons.append("bukan_bahasa_jawa")
    if TAG_RE.search(row.get("text_raw", "")):
        reasons.append("ada_tag_di_referensi")
    if re.search(r"\d", row.get("text_raw", "")):
        reasons.append("ada_angka")
    duration = float(row["duration"])
    if duration < 1.0:
        reasons.append("durasi_lt1s")
    if duration > 0 and len(row.get("text_norm", "")) / duration > 25:
        reasons.append("laju_teks_tidak_wajar")
    return reasons
