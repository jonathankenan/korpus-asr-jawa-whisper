"""Preprocessing audio untuk sistem ASR Bahasa Jawa dengan Whisper.

Langkah preprocessing yang dilakukan:
  1. Resampling ke 16 kHz (sesuai spesifikasi Fase 2 dan kebutuhan Whisper).
  2. Konversi ke mono (1 kanal) dan 16-bit PCM.
  3. Normalisasi amplitudo (peak normalization ke -3 dBFS).
  4. Voice Activity Detection (VAD) opsional menggunakan librosa untuk memangkas
     senyap di awal/akhir segmen.
  5. Pemotongan segmen dari file audio panjang sesuai kolom start/end di manifest.

Masukan : data/manifest.csv (atau salah satu fold CSV)
Keluaran : folder processed_audio/<speaker>/<utt_id>.wav

Gunakan kolom `usable == "yes"` agar segmen bermasalah (no_text, neg_duration,
beyond_audio) tidak diproses.

Jalankan dari akar repo:
  python3 src/preprocess_audio.py [--manifest data/manifest.csv] [--out processed_audio] [--vad]
"""

import argparse
import csv
import wave
import struct
import os
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGET_SR = 16_000  # Hz — spesifikasi Fase 2
TARGET_SAMPWIDTH = 2  # 16-bit
PEAK_DBFS = -3.0     # target puncak amplitudo setelah normalisasi


# ---------------------------------------------------------------------------
# Utilitas WAV minimal (tanpa scipy/soundfile agar dapat berjalan tanpa GPU)
# ---------------------------------------------------------------------------

def read_wav(path: Path):
    """Baca file WAV, kembalikan (samples_int, sample_rate, n_channels, sampwidth)."""
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        nc = wf.getnchannels()
        sw = wf.getsampwidth()
        n = wf.getnframes()
        raw = wf.readframes(n)
    fmt = {1: "b", 2: "h", 4: "i"}.get(sw)
    if fmt is None:
        raise ValueError(f"Sampwidth {sw} bytes tidak didukung: {path}")
    samples = list(struct.unpack(f"<{n * nc}{fmt}", raw))
    return samples, sr, nc, sw


def write_wav(path: Path, samples: list, sr: int):
    """Tulis samples (int16) ke file WAV mono 16-bit."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(samples)
    raw = struct.pack(f"<{n}h", *samples)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(TARGET_SAMPWIDTH)
        wf.setframerate(sr)
        wf.writeframes(raw)


def to_mono(samples: list, n_channels: int) -> list:
    """Campurkan semua kanal menjadi mono (rata-rata aritmetika)."""
    if n_channels == 1:
        return samples
    out = []
    for i in range(0, len(samples), n_channels):
        chunk = samples[i : i + n_channels]
        out.append(int(sum(chunk) / n_channels))
    return out


def resample_linear(samples: list, orig_sr: int, target_sr: int) -> list:
    """Resampling linear sederhana. Untuk kualitas lebih baik gunakan librosa."""
    if orig_sr == target_sr:
        return samples
    ratio = orig_sr / target_sr
    n_out = int(len(samples) / ratio)
    out = []
    for i in range(n_out):
        pos = i * ratio
        lo = int(pos)
        hi = min(lo + 1, len(samples) - 1)
        frac = pos - lo
        out.append(int(samples[lo] * (1 - frac) + samples[hi] * frac))
    return out


def to_int16(samples: list, orig_sampwidth: int) -> list:
    """Konversi ke 16-bit int."""
    if orig_sampwidth == 2:
        return samples
    if orig_sampwidth == 1:
        # 8-bit unsigned → 16-bit signed
        return [(s - 128) * 256 for s in samples]
    if orig_sampwidth == 4:
        # 32-bit → 16-bit (buang 16 bit rendah)
        return [s >> 16 for s in samples]
    return samples


def peak_normalize(samples: list, target_dbfs: float = PEAK_DBFS) -> list:
    """Peak normalization: skala agar nilai absolut tertinggi = target_dbfs."""
    if not samples:
        return samples
    peak = max(abs(s) for s in samples)
    if peak == 0:
        return samples
    target_linear = (32767 * 10 ** (target_dbfs / 20))
    scale = target_linear / peak
    return [max(-32768, min(32767, int(s * scale))) for s in samples]


def trim_silence(samples: list, sr: int, top_db: float = 30.0,
                 frame_ms: int = 25, hop_ms: int = 10) -> list:
    """VAD sederhana: pangkas senyap di awal/akhir berdasarkan RMS energi frame.

    Implementasi murni Python (tanpa librosa) untuk portabilitas maksimal.
    Untuk VAD berkualitas tinggi, ganti dengan librosa.effects.trim atau WebRTC VAD.
    """
    frame_len = int(sr * frame_ms / 1000)
    hop_len = int(sr * hop_ms / 1000)
    threshold_linear = 32767 * 10 ** (-top_db / 20)

    def rms(chunk):
        return math.sqrt(sum(x * x for x in chunk) / len(chunk)) if chunk else 0

    frames = []
    i = 0
    while i + frame_len <= len(samples):
        frames.append(rms(samples[i : i + frame_len]))
        i += hop_len
    if not frames:
        return samples

    start_frame = 0
    for idx, e in enumerate(frames):
        if e >= threshold_linear:
            start_frame = idx
            break

    end_frame = len(frames) - 1
    for idx in range(len(frames) - 1, -1, -1):
        if frames[idx] >= threshold_linear:
            end_frame = idx
            break

    start_sample = max(0, start_frame * hop_len)
    end_sample = min(len(samples), (end_frame + 1) * hop_len + frame_len)
    return samples[start_sample:end_sample] if end_sample > start_sample else samples


def slice_segment(samples: list, sr: int, start_sec: float, end_sec: float) -> list:
    """Potong segmen berdasarkan waktu (detik)."""
    s = max(0, int(start_sec * sr))
    e = min(len(samples), int(end_sec * sr))
    return samples[s:e]


# ---------------------------------------------------------------------------
# Pipeline utama
# ---------------------------------------------------------------------------

def process_row(row: dict, root: Path, out_dir: Path, apply_vad: bool) -> str:
    """Proses satu baris manifest. Kembalikan status ('ok', 'skip:<alasan>')."""
    if row.get("usable", "yes") != "yes":
        return "skip:not_usable"

    audio_path = root / row["audio_path"]
    if not audio_path.exists():
        return "skip:audio_missing"

    try:
        start = float(row["start"])
        end = float(row["end"])
    except (ValueError, TypeError):
        return "skip:bad_timing"

    try:
        samples, sr, nc, sw = read_wav(audio_path)
    except Exception as exc:
        return f"skip:read_error:{exc}"

    # 1. Konversi ke mono
    samples = to_mono(samples, nc)

    # 2. Konversi ke 16-bit
    samples = to_int16(samples, sw)

    # 3. Potong segmen
    samples = slice_segment(samples, sr, start, end)
    if not samples:
        return "skip:empty_slice"

    # 4. Resampling ke 16 kHz
    samples = resample_linear(samples, sr, TARGET_SR)
    sr = TARGET_SR

    # 5. Normalisasi amplitudo
    samples = peak_normalize(samples)

    # 6. VAD opsional
    if apply_vad:
        samples = trim_silence(samples, sr)

    if not samples:
        return "skip:empty_after_vad"

    utt_id = row["utt_id"]
    speaker = row["speaker"]
    out_path = out_dir / speaker / f"{utt_id}.wav"
    try:
        write_wav(out_path, samples, sr)
    except Exception as exc:
        return f"skip:write_error:{exc}"

    return "ok"


def main():
    parser = argparse.ArgumentParser(description="Preprocessing audio korpus ASR Bahasa Jawa")
    parser.add_argument("--manifest", default="data/manifest.csv",
                        help="Path ke manifest CSV (default: data/manifest.csv)")
    parser.add_argument("--out", default="processed_audio",
                        help="Folder keluaran (default: processed_audio)")
    parser.add_argument("--vad", action="store_true",
                        help="Aktifkan VAD: pangkas senyap di awal/akhir segmen")
    parser.add_argument("--limit", type=int, default=0,
                        help="Proses hanya N baris pertama (0 = semua, untuk pengujian cepat)")
    args = parser.parse_args()

    manifest_path = ROOT / args.manifest
    out_dir = ROOT / args.out

    with open(manifest_path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    if args.limit:
        rows = rows[: args.limit]

    total = len(rows)
    counts = {"ok": 0}
    skipped = []

    print(f"Memproses {total} baris dari {args.manifest} → {args.out}/")
    print(f"  Target: {TARGET_SR} Hz, mono, 16-bit, peak normalize {PEAK_DBFS} dBFS"
          + (", VAD aktif" if args.vad else ""))

    for i, row in enumerate(rows, 1):
        status = process_row(row, ROOT, out_dir, args.vad)
        if status == "ok":
            counts["ok"] += 1
        else:
            reason = status.removeprefix("skip:")
            counts[reason] = counts.get(reason, 0) + 1
            skipped.append((row.get("utt_id", "?"), reason))
        if i % 200 == 0 or i == total:
            print(f"  [{i}/{total}] ok={counts['ok']}", end="\r")

    print(f"\nSelesai. ok={counts['ok']}, dilewati={total - counts['ok']}")
    if skipped:
        print("Alasan dilewati:")
        reason_counts = {}
        for _, r in skipped:
            reason_counts[r] = reason_counts.get(r, 0) + 1
        for r, c in sorted(reason_counts.items(), key=lambda x: -x[1]):
            print(f"  {r}: {c}")

    # Tulis ringkasan ke results/
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    summary_path = results_dir / "preprocess_audio_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Manifest : {args.manifest}\n")
        f.write(f"Output   : {args.out}\n")
        f.write(f"VAD      : {'aktif' if args.vad else 'tidak aktif'}\n")
        f.write(f"Total    : {total}\n")
        for k, v in sorted(counts.items()):
            f.write(f"  {k}: {v}\n")
    print(f"Ringkasan disimpan ke {summary_path}")


if __name__ == "__main__":
    main()
