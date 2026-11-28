"""Pemuatan audio bersama untuk training, inferensi, dan preprocessing.

Kolom `audio_path` pada manifest berawalan `corpus_k11/` atau `corpus_k05/` (hasil
scripts/fetch_data.sh). Jika korpus sudah ada di folder lain, arahkan lewat variabel
lingkungan CORPUS_K11 / CORPUS_K05 tanpa perlu menyalin audio.
"""

import os
import wave
from pathlib import Path
from typing import Dict, Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TARGET_SR = 16_000
CORPUS_ENV = {"corpus_k11": "CORPUS_K11", "corpus_k05": "CORPUS_K05"}


def corpus_dir(name: str) -> Path:
    """Folder korpus `corpus_k11` / `corpus_k05`, memperhitungkan variabel lingkungan."""
    override = os.environ.get(CORPUS_ENV[name])
    if not override:
        return ROOT / name
    path = Path(override)
    return path if path.is_absolute() else ROOT / path


def resolve_audio_path(rel_path: str) -> Path:
    rel_path = rel_path.replace("\\", "/")
    head, _, tail = rel_path.partition("/")
    if head in CORPUS_ENV:
        return corpus_dir(head) / tail
    return ROOT / rel_path


def load_wav(
    audio_path: Path,
    start_sec: Optional[float] = None,
    end_sec: Optional[float] = None,
    target_sr: int = TARGET_SR,
) -> np.ndarray:
    """Baca file WAV (atau potongan [start, end]) sebagai float32 mono pada target_sr."""
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(
            f"Audio tidak ditemukan: {audio_path}. Jalankan scripts/fetch_data.sh "
            "atau set CORPUS_K11 / CORPUS_K05 ke folder korpus."
        )
    with wave.open(str(audio_path), "rb") as wf:
        sr = wf.getframerate()
        nc = wf.getnchannels()
        sw = wf.getsampwidth()
        n_frames = wf.getnframes()

        s_frame = 0 if start_sec is None else max(0, int(start_sec * sr))
        e_frame = n_frames if end_sec is None else min(n_frames, int(end_sec * sr))
        wf.setpos(min(s_frame, n_frames))
        raw = wf.readframes(max(0, e_frame - s_frame))

    if sw == 2:
        arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sw == 1:
        arr = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sw == 4:
        arr = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Sample width {sw} byte tidak didukung: {audio_path}")

    if nc > 1:
        arr = arr.reshape(-1, nc).mean(axis=1)

    if sr != target_sr and len(arr) > 0:
        n_out = int(round(len(arr) * target_sr / sr))
        x_old = np.linspace(0.0, 1.0, num=len(arr), endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
        arr = np.interp(x_new, x_old, arr).astype(np.float32)

    return arr


def peak_normalize(audio: np.ndarray, peak_dbfs: float) -> np.ndarray:
    """Skala audio agar puncak absolutnya sama dengan peak_dbfs."""
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if peak == 0.0:
        return audio
    return (audio * (10 ** (peak_dbfs / 20) / peak)).astype(np.float32)


def load_segment(
    row: Dict[str, str],
    peak_dbfs: Optional[float] = None,
    target_sr: int = TARGET_SR,
) -> np.ndarray:
    """Muat satu segmen manifest dari audio sumber, dengan normalisasi puncak opsional.

    Preprocessing diterapkan di memori; file audio korpus tidak diubah.
    """
    audio = load_wav(
        resolve_audio_path(row["audio_path"]),
        start_sec=float(row["start"]),
        end_sec=float(row["end"]),
        target_sr=target_sr,
    )
    if peak_dbfs is not None:
        audio = peak_normalize(audio, peak_dbfs)
    return audio
