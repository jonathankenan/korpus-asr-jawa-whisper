"""Skrip inferensi dan decoding ASR Bahasa Jawa dengan Whisper.

Mendukung dua mode:
  1. Mode dataset/fold: menghasilkan file CSV prediksi pada subset test/val.
  2. Mode audio tunggal (--audio): mentranskripsikan satu file WAV langsung.

Contoh:
  python src/infer.py --config configs/inference.yaml --fold 1 --dry-run
  python src/infer.py --config configs/inference.yaml --fold 1 --tag pretrained
  python src/infer.py --model models/whisper-jv-fold1 --audio sample.wav
"""

import argparse
import csv
import re
import struct
import time
import unicodedata
import wave
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import yaml
from transformers import WhisperForConditionalGeneration, WhisperProcessor

try:
    from normalize_text import normalize as normalize_text
except Exception:
    def normalize_text(text: str) -> str:
        text = unicodedata.normalize("NFKD", text or "")
        text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
        text = re.sub(r"\[[^\]]*\]", " ", text).lower()
        text = re.sub(r"[^a-z0-9'\- ]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

ROOT = Path(__file__).resolve().parent.parent


def load_wav(
    audio_path: Path,
    start_sec: Optional[float] = None,
    end_sec: Optional[float] = None,
    target_sr: int = 16_000,
) -> np.ndarray:
    """Baca file WAV (atau segmen tertentu) sebagai float32 mono 16 kHz."""
    with wave.open(str(audio_path), "rb") as wf:
        sr = wf.getframerate()
        nc = wf.getnchannels()
        sw = wf.getsampwidth()
        n_frames = wf.getnframes()

        s_frame = max(0, int(start_sec * sr)) if start_sec is not None else 0
        e_frame = min(n_frames, int(end_sec * sr)) if end_sec is not None else n_frames
        wf.setpos(s_frame)
        raw = wf.readframes(max(0, e_frame - s_frame))

    if not raw:
        return np.zeros(0, dtype=np.float32)

    if sw == 2:
        arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sw == 1:
        arr = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sw == 4:
        arr = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        fmt = f"<{len(raw) // sw}h"
        arr = np.array(struct.unpack(fmt, raw), dtype=np.float32) / 32768.0

    if nc > 1:
        arr = arr.reshape(-1, nc).mean(axis=1)

    if sr != target_sr and len(arr) > 0:
        n_out = int(round(len(arr) * target_sr / sr))
        x_old = np.linspace(0.0, 1.0, num=len(arr), endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
        arr = np.interp(x_new, x_old, arr).astype(np.float32)

    return arr


def transcribe_array(
    audio: np.ndarray,
    processor: WhisperProcessor,
    model: WhisperForConditionalGeneration,
    device: str,
    sr: int = 16_000,
    chunk_sec: float = 30.0,
    num_beams: int = 5,
    max_new_tokens: int = 225,
    no_repeat_ngram_size: int = 3,
    temperature: float = 0.0,
) -> str:
    """Dekodekan array audio (dengan pemotongan otomatis jika > chunk_sec)."""
    if len(audio) == 0:
        return ""

    chunk_samples = int(chunk_sec * sr)
    texts: List[str] = []

    for start in range(0, len(audio), chunk_samples):
        chunk = audio[start : start + chunk_samples]
        inputs = processor(chunk, sampling_rate=sr, return_tensors="pt")
        input_features = inputs.input_features.to(device)

        with torch.no_grad():
            predicted_ids = model.generate(
                input_features,
                num_beams=num_beams,
                max_new_tokens=max_new_tokens,
                no_repeat_ngram_size=no_repeat_ngram_size,
                temperature=temperature,
            )
        decoded = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
        if decoded:
            texts.append(decoded)

    return " ".join(texts).strip()


def resolve_device(pref: str) -> str:
    if pref == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return pref


def load_split_rows(manifest_path: Path, split_path: Path, split_name: str) -> List[Dict[str, str]]:
    with open(split_path, encoding="utf-8", newline="") as f:
        split_map = {r["utt_id"]: r["split"] for r in csv.DictReader(f)}

    rows: List[Dict[str, str]] = []
    with open(manifest_path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("usable", "yes") != "yes":
                continue
            if split_map.get(row["utt_id"]) == split_name:
                rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Inferensi Whisper ASR Bahasa Jawa")
    parser.add_argument("--config", default="configs/inference.yaml")
    parser.add_argument("--model", default=None, help="Override nama/path model")
    parser.add_argument("--fold", type=int, default=1, choices=[1, 2, 3, 4, 5])
    parser.add_argument("--split", default=None, choices=["train", "val", "test"])
    parser.add_argument("--audio", default=None, help="Path satu file WAV untuk inferensi tunggal")
    parser.add_argument("--tag", default=None, help="Tag nama file keluaran")
    parser.add_argument("--num-beams", type=int, default=None)
    parser.add_argument("--limit", type=int, default=0, help="Batasi N segmen (0 = semua)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Verifikasi konfigurasi & data tanpa memuat bobot model")
    args = parser.parse_args()

    with open(ROOT / args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    m_cfg = cfg["model"]
    dec_cfg = cfg["decoding"]
    d_cfg = cfg["data"]
    o_cfg = cfg["output"]

    model_id = args.model or m_cfg["model_name_or_path"]
    device = resolve_device(m_cfg.get("device", "auto"))
    num_beams = args.num_beams or dec_cfg["num_beams"]
    split_name = args.split or d_cfg["split_name"]
    tag = args.tag or o_cfg["tag"]

    if args.audio:
        audio_path = Path(args.audio)
        if not audio_path.is_absolute():
            audio_path = ROOT / audio_path
        print(f"Mode audio tunggal: {audio_path} | Model: {model_id} | Device: {device}")
        if args.dry_run:
            print("Dry-run selesai.")
            return
        processor = WhisperProcessor.from_pretrained(
            model_id, language=m_cfg["language"], task=m_cfg["task"]
        )
        model = WhisperForConditionalGeneration.from_pretrained(model_id).to(device)
        model.eval()
        model.generation_config.language = m_cfg["language"]
        model.generation_config.task = m_cfg["task"]
        model.generation_config.forced_decoder_ids = None

        audio = load_wav(audio_path, target_sr=d_cfg["sample_rate"])
        text = transcribe_array(
            audio, processor, model, device,
            sr=d_cfg["sample_rate"],
            chunk_sec=d_cfg["chunk_length_sec"],
            num_beams=num_beams,
            max_new_tokens=dec_cfg["max_new_tokens"],
            no_repeat_ngram_size=dec_cfg["no_repeat_ngram_size"],
            temperature=dec_cfg["temperature"],
        )
        print(f"Transkripsi     : {text}")
        print(f"Normalisasi     : {normalize_text(text)}")
        return

    manifest_path = ROOT / d_cfg["manifest_path"]
    if not manifest_path.exists():
        manifest_path = ROOT / d_cfg["fallback_manifest"]
    split_path = ROOT / d_cfg["splits_dir"] / f"fold{args.fold}.csv"
    rows = load_split_rows(manifest_path, split_path, split_name)
    if args.limit:
        rows = rows[: args.limit]

    out_dir = ROOT / o_cfg["results_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"predictions_fold{args.fold}_{tag}.csv"

    print(f"Fold {args.fold} ({split_name}) | {len(rows)} segmen | Model: {model_id} | beams={num_beams}")
    print(f"Keluaran: {out_csv}")
    if args.dry_run:
        print("Dry-run selesai: data dan konfigurasi inferensi valid.")
        return

    processor = WhisperProcessor.from_pretrained(
        model_id, language=m_cfg["language"], task=m_cfg["task"]
    )
    model = WhisperForConditionalGeneration.from_pretrained(model_id).to(device)
    model.eval()
    model.generation_config.language = m_cfg["language"]
    model.generation_config.task = m_cfg["task"]
    model.generation_config.forced_decoder_ids = None

    proc_dir = ROOT / d_cfg["processed_audio_dir"]
    results = []
    t0 = time.time()

    for idx, row in enumerate(rows, 1):
        utt_id = row["utt_id"]
        spk = row["speaker"]
        pre_wav = proc_dir / spk / f"{utt_id}.wav"
        if pre_wav.exists():
            audio = load_wav(pre_wav, target_sr=d_cfg["sample_rate"])
        else:
            audio = load_wav(
                ROOT / row["audio_path"],
                start_sec=float(row["start"]),
                end_sec=float(row["end"]),
                target_sr=d_cfg["sample_rate"],
            )

        pred_raw = transcribe_array(
            audio, processor, model, device,
            sr=d_cfg["sample_rate"],
            chunk_sec=d_cfg["chunk_length_sec"],
            num_beams=num_beams,
            max_new_tokens=dec_cfg["max_new_tokens"],
            no_repeat_ngram_size=dec_cfg["no_repeat_ngram_size"],
            temperature=dec_cfg["temperature"],
        )
        ref_raw = row.get("text_raw", "")
        ref_norm = row.get("text_norm") or normalize_text(ref_raw)
        pred_norm = normalize_text(pred_raw)

        results.append({
            "utt_id": utt_id,
            "speaker": spk,
            "group": row.get("group", ""),
            "dialect": row.get("dialect", ""),
            "speaker_status": row.get("speaker_status", ""),
            "duration": row.get("duration", ""),
            "ref_raw": ref_raw,
            "ref_norm": ref_norm,
            "pred_raw": pred_raw,
            "pred_norm": pred_norm,
        })
        if idx % 25 == 0 or idx == len(rows):
            elapsed = time.time() - t0
            print(f"  [{idx}/{len(rows)}] {elapsed:.1f}s", end="\r")

    fieldnames = list(results[0].keys()) if results else []
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(results)
    print(f"\nSelesai: {len(results)} prediksi disimpan ke {out_csv}")


if __name__ == "__main__":
    main()
