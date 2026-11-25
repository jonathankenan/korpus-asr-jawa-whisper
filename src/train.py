"""Fine-tuning model Whisper untuk ASR Bahasa Jawa (5-fold speaker-independent).

Contoh penggunaan:
  python src/train.py --config configs/train.yaml --fold 1
  python src/train.py --fold 1 --model openai/whisper-small --epochs 3 --lr 1e-5
  python src/train.py --fold 1 --dry-run
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

from model import DataCollatorSpeechSeq2SeqWithPadding, load_wav_segment, load_whisper

ROOT = Path(__file__).resolve().parent.parent


def _levenshtein(a: List[str], b: List[str]) -> int:
    """Hitung jarak edit Levenshtein antara dua urutan token."""
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev = dp[0]
        dp[0] = i
        for j, cb in enumerate(b, 1):
            cur = dp[j]
            dp[j] = prev if ca == cb else 1 + min(prev, dp[j], dp[j - 1])
            prev = cur
    return dp[-1]


def compute_wer(refs: List[str], hyps: List[str]) -> float:
    """Hitung Word Error Rate (WER), pakai jiwer jika tersedia."""
    try:
        import jiwer
        return float(jiwer.wer(refs, hyps))
    except ImportError:
        err, words = 0, 0
        for r, h in zip(refs, hyps):
            rw, hw = r.split(), h.split()
            err += _levenshtein(rw, hw)
            words += len(rw)
        return err / max(1, words)


class JavaneseSpeechDataset(Dataset):
    """Dataset segmen ucapan Bahasa Jawa untuk Seq2SeqTrainer."""

    def __init__(
        self,
        rows: List[Dict[str, str]],
        processor,
        text_col: str = "text_norm",
        processed_dir: Path = ROOT / "processed_audio",
        sample_rate: int = 16_000,
    ):
        self.rows = rows
        self.processor = processor
        self.text_col = text_col
        self.processed_dir = processed_dir
        self.sample_rate = sample_rate

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        row = self.rows[idx]
        utt_id = row["utt_id"]
        spk = row["speaker"]
        pre_wav = self.processed_dir / spk / f"{utt_id}.wav"

        if pre_wav.exists():
            audio = load_wav_segment(pre_wav, target_sr=self.sample_rate)
        else:
            audio = load_wav_segment(
                ROOT / row["audio_path"],
                start_sec=float(row["start"]),
                end_sec=float(row["end"]),
                target_sr=self.sample_rate,
            )

        features = self.processor.feature_extractor(
            audio, sampling_rate=self.sample_rate
        ).input_features[0]
        text = (row.get(self.text_col) or row.get("text_raw") or "").strip()
        labels = self.processor.tokenizer(text).input_ids
        return {"input_features": features, "labels": labels}


def load_fold_rows(
    manifest_path: Path,
    split_path: Path,
    text_col: str,
    min_dur: float,
    max_dur: float,
) -> Dict[str, List[Dict[str, str]]]:
    """Muat baris manifest yang usable dan kelompokkan menjadi train/val/test."""
    with open(split_path, encoding="utf-8", newline="") as f:
        split_map = {r["utt_id"]: r["split"] for r in csv.DictReader(f)}

    subsets: Dict[str, List[Dict[str, str]]] = {"train": [], "val": [], "test": []}
    with open(manifest_path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("usable", "yes") != "yes":
                continue
            text = (row.get(text_col) or row.get("text_raw") or "").strip()
            if not text:
                continue
            try:
                dur = float(row["duration"])
            except (ValueError, TypeError):
                continue
            if dur < min_dur or dur > max_dur:
                continue
            sp = split_map.get(row["utt_id"])
            if sp in subsets:
                subsets[sp].append(row)
    return subsets


def main():
    parser = argparse.ArgumentParser(description="Fine-tuning Whisper ASR Bahasa Jawa")
    parser.add_argument("--config", default="configs/train.yaml", help="Path file YAML config")
    parser.add_argument("--fold", type=int, default=1, choices=[1, 2, 3, 4, 5])
    parser.add_argument("--model", default=None, help="Override nama model pretrained")
    parser.add_argument("--manifest", default=None, help="Override path manifest CSV")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="Verifikasi konfigurasi & jumlah data tanpa training")
    args = parser.parse_args()

    with open(ROOT / args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    m_cfg = cfg["model"]
    d_cfg = cfg["data"]
    t_cfg = cfg["training"]

    model_name = args.model or m_cfg["pretrained_name"]
    epochs = args.epochs or t_cfg["num_train_epochs"]
    batch_size = args.batch_size or t_cfg["per_device_train_batch_size"]
    lr = args.lr or t_cfg["learning_rate"]
    base_out = args.output_dir or t_cfg["output_dir"]
    out_dir = ROOT / f"{base_out}-fold{args.fold}"

    manifest_rel = args.manifest or d_cfg["manifest_path"]
    manifest_path = ROOT / manifest_rel
    if not manifest_path.exists():
        manifest_path = ROOT / d_cfg["fallback_manifest"]

    split_path = ROOT / d_cfg["splits_dir"] / f"fold{args.fold}.csv"
    subsets = load_fold_rows(
        manifest_path=manifest_path,
        split_path=split_path,
        text_col=d_cfg["text_column"],
        min_dur=d_cfg["min_duration_sec"],
        max_dur=d_cfg["max_duration_sec"],
    )

    print(f"Fold {args.fold} | Model: {model_name} | Manifest: {manifest_path.name}")
    print(f"  Train: {len(subsets['train'])} | Val: {len(subsets['val'])} | Test: {len(subsets['test'])}")
    print(f"  Output: {out_dir}")

    if args.dry_run:
        print("Dry-run selesai: data dan konfigurasi valid.")
        return

    processor, model = load_whisper(
        model_name=model_name,
        language=m_cfg["language"],
        task=m_cfg["task"],
        freeze_feature_encoder=m_cfg.get("freeze_feature_encoder", False),
    )

    proc_dir = ROOT / d_cfg["processed_audio_dir"]
    train_ds = JavaneseSpeechDataset(subsets["train"], processor, d_cfg["text_column"], proc_dir, d_cfg["sample_rate"])
    val_ds = JavaneseSpeechDataset(subsets["val"], processor, d_cfg["text_column"], proc_dir, d_cfg["sample_rate"])

    collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        return {"wer": round(compute_wer(label_str, pred_str) * 100, 2)}

    use_fp16 = bool(t_cfg.get("fp16", True) and torch.cuda.is_available())
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=t_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=t_cfg["gradient_accumulation_steps"],
        learning_rate=lr,
        warmup_steps=t_cfg["warmup_steps"],
        weight_decay=t_cfg["weight_decay"],
        fp16=use_fp16,
        gradient_checkpointing=t_cfg.get("gradient_checkpointing", True) and torch.cuda.is_available(),
        eval_strategy=t_cfg["eval_strategy"],
        save_strategy=t_cfg["save_strategy"],
        logging_steps=t_cfg["logging_steps"],
        save_total_limit=t_cfg["save_total_limit"],
        predict_with_generate=True,
        generation_max_length=t_cfg["generation_max_length"],
        load_best_model_at_end=t_cfg["load_best_model_at_end"],
        metric_for_best_model=t_cfg["metric_for_best_model"],
        greater_is_better=t_cfg["greater_is_better"],
        seed=t_cfg["seed"],
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
        processing_class=processor.feature_extractor,
    )

    train_result = trainer.train()
    trainer.save_model(str(out_dir))
    processor.save_pretrained(str(out_dir))

    metrics_path = out_dir / "train_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(train_result.metrics, f, indent=2)
    print(f"Pelatihan selesai. Model & metrik disimpan di {out_dir}")


if __name__ == "__main__":
    main()
