"""Inferensi dan decoding ASR Bahasa Jawa dengan Whisper.

Mode:
  1. Dataset  : menulis prediksi ke results/predictions/<exp>/fold<k>.csv.
                `--fold all` mentranskripsi test set kelima fold sekaligus (dipakai untuk
                model pretrained, karena test set antar fold tidak beririsan).
  2. Satu file: `--audio file.wav` mencetak transkripsi.

Proses dapat dilanjutkan: segmen yang sudah ada di CSV dilewati. Konfigurasi tiap file
prediksi dicatat di fold<k>.json; menjalankan ulang dengan konfigurasi berbeda ditolak
kecuali memakai --overwrite.

Contoh:
  python src/infer.py --exp tiny-pretrained --model openai/whisper-tiny --fold all
  python src/infer.py --exp tiny-ft --model models/tiny-ft/fold1 --fold 1
  python src/infer.py --model openai/whisper-small --audio sample.wav
"""

import argparse
import csv
import json
import platform
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import transformers
import yaml

from audio import load_segment, load_wav, peak_normalize
from data import FOLDS, load_manifest, split_map, test_fold_by_speaker
from model import load_whisper, resolve_model
from normalize_text import normalize

ROOT = Path(__file__).resolve().parent.parent
FIELDS = ["utt_id", "speaker", "fold", "split", "pred_raw", "pred_norm"]


def transcribe_batch(chunks: List[np.ndarray], processor, model, device: str,
                     gen_kwargs: Dict, sample_rate: int) -> List[str]:
    """Transkripsikan sekumpulan potongan audio (masing-masing paling lama 30 detik)."""
    inputs = processor.feature_extractor(
        chunks, sampling_rate=sample_rate, return_tensors="pt", return_attention_mask=True
    )
    with torch.inference_mode():
        ids = model.generate(
            inputs.input_features.to(device),
            attention_mask=inputs.attention_mask.to(device),
            **gen_kwargs,
        )
    return [t.strip() for t in processor.tokenizer.batch_decode(ids, skip_special_tokens=True)]


def transcribe_many(audios: List[np.ndarray], processor, model, device: str, gen_kwargs: Dict,
                    sample_rate: int = 16_000, chunk_sec: float = 30.0) -> List[str]:
    """Transkripsikan beberapa audio; audio lebih panjang dari chunk_sec dipotong berurutan."""
    size = int(chunk_sec * sample_rate)
    chunks, owner = [], []
    for i, audio in enumerate(audios):
        for start in range(0, len(audio), size):
            chunks.append(audio[start:start + size])
            owner.append(i)
    texts: List[List[str]] = [[] for _ in audios]
    if chunks:
        for i, text in zip(owner, transcribe_batch(chunks, processor, model, device,
                                                    gen_kwargs, sample_rate)):
            if text:
                texts[i].append(text)
    return [" ".join(t) for t in texts]


def parse_peak(value, default):
    if value is None:
        return default
    return None if str(value).lower() in ("none", "off") else float(value)


def main():
    parser = argparse.ArgumentParser(description="Inferensi Whisper ASR Bahasa Jawa")
    parser.add_argument("--config", default="configs/inference.yaml")
    parser.add_argument("--model", default=None, help="Nama Hugging Face atau folder model")
    parser.add_argument("--exp", default=None, help="Nama eksperimen (folder keluaran)")
    parser.add_argument("--fold", default="all", help="1-5 atau all")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--audio", default=None, help="Satu file WAV untuk inferensi tunggal")
    parser.add_argument("--num-beams", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--peak-dbfs", default=None, help="Angka dBFS, atau 'none'")
    parser.add_argument("--limit", type=int, default=0, help="Batasi N segmen per fold (uji cepat)")
    parser.add_argument("--overwrite", action="store_true", help="Buang prediksi lama")
    parser.add_argument("--dry-run", action="store_true", help="Cek data tanpa memuat model")
    args = parser.parse_args()

    with open(ROOT / args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    m_cfg, dec, a_cfg = cfg["model"], cfg["decoding"], cfg["audio"]

    model_name = args.model or m_cfg["model_name_or_path"]
    device = ("cuda" if torch.cuda.is_available() else "cpu") if m_cfg["device"] == "auto" else m_cfg["device"]
    num_beams = args.num_beams or dec["num_beams"]
    batch_size = args.batch_size or cfg["data"]["batch_size"]
    peak_dbfs = parse_peak(args.peak_dbfs, a_cfg["peak_dbfs"])
    sample_rate, chunk_sec = a_cfg["sample_rate"], a_cfg["chunk_length_sec"]
    gen_kwargs = {
        "language": m_cfg["language"],
        "task": m_cfg["task"],
        "num_beams": num_beams,
        "do_sample": False,
        "max_new_tokens": dec["max_new_tokens"],
        "no_repeat_ngram_size": dec["no_repeat_ngram_size"],
    }

    if args.audio:
        processor, model = load_whisper(model_name, m_cfg["language"], m_cfg["task"])
        model.to(device).eval()
        audio = load_wav(Path(args.audio), target_sr=sample_rate)
        if peak_dbfs is not None:
            audio = peak_normalize(audio, peak_dbfs)
        text = transcribe_many([audio], processor, model, device, gen_kwargs, sample_rate, chunk_sec)[0]
        print(f"Transkripsi : {text}")
        print(f"Normalisasi : {normalize(text)}")
        return

    if not args.exp:
        raise SystemExit("--exp wajib diisi untuk inferensi dataset.")

    manifest = load_manifest()
    if args.fold == "all":
        if args.split != "test":
            raise SystemExit("--fold all hanya untuk --split test.")
        fold_of = test_fold_by_speaker()
        jobs = {k: [r for r in manifest if fold_of[r["speaker"]] == k] for k in FOLDS}
    else:
        k = int(args.fold)
        smap = split_map(k)
        jobs = {k: [r for r in manifest if smap.get(r["utt_id"]) == args.split]}

    out_dir = ROOT / cfg["output"]["predictions_dir"] / args.exp
    revision = resolve_model(model_name)[1].get("revision", "")
    processor = model = None

    for fold, rows in jobs.items():
        if args.limit:
            rows = rows[:args.limit]
        out_csv = out_dir / f"fold{fold}.csv"
        meta_path = out_dir / f"fold{fold}.json"
        signature = {
            "model": Path(model_name).as_posix(), "revision": revision, "split": args.split,
            "num_beams": num_beams, "max_new_tokens": dec["max_new_tokens"],
            "no_repeat_ngram_size": dec["no_repeat_ngram_size"], "peak_dbfs": peak_dbfs,
            "chunk_length_sec": chunk_sec, "language": m_cfg["language"], "task": m_cfg["task"],
        }
        done, meta = set(), {}
        if out_csv.exists() and not args.overwrite:
            meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
            if meta.get("signature") != signature:
                raise SystemExit(f"{out_csv} dibuat dengan konfigurasi lain. Pakai --exp lain atau --overwrite.")
            with open(out_csv, encoding="utf-8", newline="") as f:
                done = {r["utt_id"] for r in csv.DictReader(f)}
        todo = [r for r in rows if r["utt_id"] not in done]
        print(f"[{args.exp}] fold {fold} ({args.split}): {len(rows)} segmen, "
              f"{len(done)} sudah ada, {len(todo)} dikerjakan", flush=True)
        if args.dry_run or not todo:
            continue

        if model is None:
            processor, model = load_whisper(model_name, m_cfg["language"], m_cfg["task"])
            model.to(device).eval()

        out_dir.mkdir(parents=True, exist_ok=True)
        todo.sort(key=lambda r: float(r["duration"]))  # durasi mirip -> decoding batch lebih cepat
        audio_sec = meta.get("audio_seconds", 0.0)
        wall_sec = meta.get("wall_seconds", 0.0)
        append = bool(done)
        with open(out_csv, "a" if append else "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            if not append:
                writer.writeheader()
            t_fold = time.time()
            for i in range(0, len(todo), batch_size):
                batch = todo[i:i + batch_size]
                t0 = time.time()
                audios = [load_segment(r, peak_dbfs, sample_rate) for r in batch]
                preds = transcribe_many(audios, processor, model, device, gen_kwargs,
                                        sample_rate, chunk_sec)
                wall_sec += time.time() - t0
                audio_sec += sum(len(a) for a in audios) / sample_rate
                for row, pred in zip(batch, preds):
                    writer.writerow({
                        "utt_id": row["utt_id"], "speaker": row["speaker"], "fold": fold,
                        "split": args.split, "pred_raw": pred, "pred_norm": normalize(pred),
                    })
                f.flush()
                meta = {
                    "signature": signature,
                    "audio_seconds": round(audio_sec, 1),
                    "wall_seconds": round(wall_sec, 1),
                    "batch_size": batch_size,
                    "device": device if device == "cpu" else torch.cuda.get_device_name(0),
                    "machine": platform.processor(),
                    "torch": torch.__version__,
                    "transformers": transformers.__version__,
                }
                meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
                n = min(i + batch_size, len(todo))
                if n == len(todo) or (i // batch_size) % 10 == 0:
                    rate = n / (time.time() - t_fold)
                    print(f"    {n}/{len(todo)} segmen | {rate:.2f} segmen/s | "
                          f"sisa ~{(len(todo) - n) / rate / 60:.0f} menit", flush=True)
        print(f"  selesai: {out_csv}", flush=True)


if __name__ == "__main__":
    main()
