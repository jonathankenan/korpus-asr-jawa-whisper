"""Fine-tuning Whisper untuk ASR Bahasa Jawa (5-fold antar penutur).

Keluaran:
  models/<exp>/fold<k>/                 model terbaik menurut WER validation
  results/training/<exp>/fold<k>.json   konfigurasi, riwayat loss/WER, dan waktu latih

Fold yang sudah selesai dengan konfigurasi sama akan dilewati, sehingga proses yang
terputus bisa diulang dengan perintah yang sama.

Contoh:
  python src/train.py --exp small-ft --model openai/whisper-small --fold 1 --epochs 5
  python src/train.py --exp tiny-ft --model openai/whisper-tiny --freeze-encoder --fold 1
  python src/train.py --fold 1 --dry-run
"""

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import transformers
import yaml
from torch.utils.data import Dataset
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, set_seed

from audio import load_segment
from data import load_manifest, split_map
from evaluate import compute_corpus_metrics
from model import DataCollatorSpeechSeq2SeqWithPadding, load_whisper, resolve_model
from normalize_text import normalize

ROOT = Path(__file__).resolve().parent.parent
MAX_LABEL_TOKENS = 448  # batas posisi decoder Whisper


class JavaneseSpeechDataset(Dataset):
    """Segmen ucapan: fitur log-Mel dihitung saat diambil, audio sumber tidak diubah."""

    def __init__(self, rows: List[Dict[str, str]], processor, peak_dbfs: Optional[float],
                 sample_rate: int = 16_000, text_col: str = "text_norm"):
        self.rows = rows
        self.processor = processor
        self.peak_dbfs = peak_dbfs
        self.sample_rate = sample_rate
        self.text_col = text_col

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        row = self.rows[idx]
        audio = load_segment(row, self.peak_dbfs, self.sample_rate)
        features = self.processor.feature_extractor(
            audio, sampling_rate=self.sample_rate
        ).input_features[0]
        labels = self.processor.tokenizer(
            row[self.text_col], truncation=True, max_length=MAX_LABEL_TOKENS
        ).input_ids
        return {"input_features": features, "labels": labels}


def load_fold_rows(fold: int, text_col: str, min_dur: float,
                   max_dur: float) -> Dict[str, List[Dict[str, str]]]:
    """Baris usable pada satu fold, dikelompokkan menjadi train/val/test.

    Filter durasi hanya berlaku untuk train dan validation; test dievaluasi utuh oleh infer.py.
    """
    smap = split_map(fold)
    subsets: Dict[str, List[Dict[str, str]]] = {"train": [], "val": [], "test": []}
    for row in load_manifest():
        split = smap.get(row["utt_id"])
        if split not in subsets or not row.get(text_col, "").strip():
            continue
        if split != "test" and not min_dur <= float(row["duration"]) <= max_dur:
            continue
        subsets[split].append(row)
    return subsets


def subsample(rows: List[Dict[str, str]], n: int) -> List[Dict[str, str]]:
    """Ambil n baris berjarak sama (untuk uji cepat)."""
    if not n or n >= len(rows):
        return rows
    return [rows[int(i * len(rows) / n)] for i in range(n)]


def main():
    parser = argparse.ArgumentParser(description="Fine-tuning Whisper ASR Bahasa Jawa")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--exp", default=None, help="Nama eksperimen")
    parser.add_argument("--fold", type=int, default=1, choices=[1, 2, 3, 4, 5])
    parser.add_argument("--model", default=None, help="Override nama model pretrained")
    parser.add_argument("--freeze-encoder", action="store_true", help="Latih decoder saja")
    parser.add_argument("--epochs", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--max-train", type=int, default=0, help="Batasi data latih (uji cepat)")
    parser.add_argument("--max-val", type=int, default=0, help="Batasi data validation (uji cepat)")
    parser.add_argument("--overwrite", action="store_true", help="Latih ulang walau sudah ada")
    parser.add_argument("--dry-run", action="store_true", help="Cek konfigurasi & jumlah data")
    args = parser.parse_args()

    with open(ROOT / args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    m_cfg, d_cfg, t_cfg = cfg["model"], cfg["data"], cfg["training"]

    def pick(value, default):
        return default if value is None else value

    model_name = args.model or m_cfg["pretrained_name"]
    freeze = args.freeze_encoder or m_cfg.get("freeze_encoder", False)
    exp = args.exp or model_name.rsplit("-", 1)[-1] + ("-ft-frozen" if freeze else "-ft")
    out_dir = ROOT / t_cfg["models_dir"] / exp / f"fold{args.fold}"
    record_path = ROOT / t_cfg["records_dir"] / exp / f"fold{args.fold}.json"

    settings = {
        "model": model_name,
        "revision": resolve_model(model_name)[1].get("revision", ""),
        "freeze_encoder": freeze,
        "epochs": pick(args.epochs, t_cfg["num_train_epochs"]),
        "batch_size": pick(args.batch_size, t_cfg["per_device_train_batch_size"]),
        "gradient_accumulation_steps": pick(args.grad_accum, t_cfg["gradient_accumulation_steps"]),
        "learning_rate": float(pick(args.lr, t_cfg["learning_rate"])),
        "warmup_steps": pick(args.warmup_steps, t_cfg["warmup_steps"]),
        "weight_decay": t_cfg["weight_decay"],
        "seed": t_cfg["seed"],
        "peak_dbfs": d_cfg["peak_dbfs"],
        "min_duration_sec": d_cfg["min_duration_sec"],
        "max_duration_sec": d_cfg["max_duration_sec"],
        "max_train": args.max_train,
        "max_val": args.max_val,
        "val_no_repeat_ngram_size": t_cfg["generation_no_repeat_ngram_size"],
    }

    subsets = load_fold_rows(args.fold, d_cfg["text_column"],
                             d_cfg["min_duration_sec"], d_cfg["max_duration_sec"])
    train_rows = subsample(subsets["train"], args.max_train)
    val_rows = subsample(subsets["val"], args.max_val)

    print(f"[{exp}] fold {args.fold} | {model_name} | encoder {'beku' if freeze else 'dilatih'} "
          f"| lr {settings['learning_rate']} | epoch {settings['epochs']}", flush=True)
    print(f"  train: {len(train_rows)} | val: {len(val_rows)} | test: {len(subsets['test'])}")
    print(f"  model -> {out_dir}")

    if args.dry_run:
        print("Dry-run selesai: data dan konfigurasi valid.")
        return

    if record_path.exists() and (out_dir / "model.safetensors").exists() and not args.overwrite:
        old = json.loads(record_path.read_text(encoding="utf-8"))
        if old["settings"] != settings:
            raise SystemExit(f"{out_dir} dilatih dengan konfigurasi lain. Pakai --exp lain atau --overwrite.")
        print("  sudah selesai sebelumnya, dilewati.")
        return

    set_seed(settings["seed"])
    processor, model = load_whisper(model_name, m_cfg["language"], m_cfg["task"],
                                    freeze_encoder=freeze, for_training=True)
    # decoding validation disamakan dengan inferensi (configs/inference.yaml)
    model.generation_config.no_repeat_ngram_size = settings["val_no_repeat_ngram_size"]
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    def make_dataset(rows):
        return JavaneseSpeechDataset(rows, processor, d_cfg["peak_dbfs"],
                                     d_cfg["sample_rate"], d_cfg["text_column"])

    collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    def compute_metrics(pred):
        pad = processor.tokenizer.pad_token_id
        pred_ids = np.where(pred.predictions != -100, pred.predictions, pad)
        label_ids = np.where(pred.label_ids != -100, pred.label_ids, pad)
        hyps = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        refs = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        pairs = [{"ref": normalize(r), "hyp": normalize(h)} for r, h in zip(refs, hyps)]
        metrics = compute_corpus_metrics(pairs, "ref", "hyp")
        return {"wer": metrics["wer_pct"], "cer": metrics["cer_pct"]}

    cuda = torch.cuda.is_available()
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=settings["epochs"],
        per_device_train_batch_size=settings["batch_size"],
        per_device_eval_batch_size=t_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=settings["gradient_accumulation_steps"],
        learning_rate=settings["learning_rate"],
        warmup_steps=settings["warmup_steps"],
        weight_decay=settings["weight_decay"],
        fp16=bool(t_cfg["fp16"] and cuda),
        gradient_checkpointing=bool(t_cfg["gradient_checkpointing"] and cuda and not freeze),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        save_only_model=True,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        logging_steps=t_cfg["logging_steps"],
        predict_with_generate=True,
        generation_max_length=t_cfg["generation_max_length"],
        dataloader_num_workers=t_cfg["dataloader_num_workers"] if cuda else 0,
        remove_unused_columns=False,
        disable_tqdm=True,
        dataloader_pin_memory=cuda,
        seed=settings["seed"],
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=make_dataset(train_rows),
        eval_dataset=make_dataset(val_rows),
        data_collator=collator,
        compute_metrics=compute_metrics,
        processing_class=processor.feature_extractor,
    )

    started = datetime.now()
    before = trainer.evaluate()
    print(f"  validation sebelum latih: WER {before['eval_wer']}% | CER {before['eval_cer']}%", flush=True)
    result = trainer.train()
    evals = [h for h in trainer.state.log_history if "eval_wer" in h and h.get("epoch", 0) > 0]
    after = min(evals, key=lambda h: h["eval_wer"])  # epoch yang dimuat ulang sebagai model terbaik
    print(f"  validation model terbaik (epoch {after['epoch']:g}): WER {after['eval_wer']}% "
          f"| CER {after['eval_cer']}%", flush=True)

    trainer.save_model(str(out_dir))
    processor.save_pretrained(str(out_dir))
    for checkpoint in out_dir.glob("checkpoint-*"):
        shutil.rmtree(checkpoint, ignore_errors=True)

    record = {
        "exp": exp,
        "fold": args.fold,
        "settings": settings,
        "n_train": len(train_rows),
        "n_val": len(val_rows),
        "trainable_parameters": trainable,
        "total_parameters": total,
        "val_before": {"wer": before["eval_wer"], "cer": before["eval_cer"], "loss": before["eval_loss"]},
        "val_best": {"wer": after["eval_wer"], "cer": after["eval_cer"], "loss": after["eval_loss"],
                     "epoch": after["epoch"]},
        "history": trainer.state.log_history,
        "train_metrics": result.metrics,
        "started": started.isoformat(timespec="seconds"),
        "wall_seconds": round((datetime.now() - started).total_seconds(), 1),
        "device": torch.cuda.get_device_name(0) if cuda else "cpu",
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"  selesai dalam {record['wall_seconds'] / 60:.1f} menit. Catatan: {record_path}", flush=True)


if __name__ == "__main__":
    main()
