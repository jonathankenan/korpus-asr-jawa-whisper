"""Utilitas model Whisper dan data collator untuk ASR Bahasa Jawa.

Referensi:
  - Radford et al. (2022), "Robust Speech Recognition via Large-Scale Weak Supervision"
  - Hugging Face Transformers: WhisperForConditionalGeneration & WhisperProcessor
    https://huggingface.co/docs/transformers/model_doc/whisper
  - DataCollatorSpeechSeq2SeqWithPadding diadaptasi dari S. Gandhi (2022),
    "Fine-Tune Whisper For Multilingual ASR with Transformers"
    https://huggingface.co/blog/fine-tune-whisper
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import yaml
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers.utils import logging as hf_logging

ROOT = Path(__file__).resolve().parent.parent

# Peringatan generate() Whisper di transformers 5 berulang tiap batch dan tidak memengaruhi hasil
hf_logging.set_verbosity_error()
hf_logging.disable_progress_bar()


def resolve_model(name: str) -> Tuple[str, Dict[str, str]]:
    """Terjemahkan nama model menjadi argumen from_pretrained.

    Folder lokal (hasil fine-tuning) dipakai apa adanya; nama Hugging Face dikunci pada
    revisi di configs/models.yaml.
    """
    for candidate in (Path(name), ROOT / name):
        if candidate.is_dir():
            return str(candidate), {}
    with open(ROOT / "configs" / "models.yaml", encoding="utf-8") as f:
        known = yaml.safe_load(f)["models"]
    if name in known:
        return name, {"revision": known[name]["revision"]}
    if name.startswith("openai/"):
        return name, {}
    raise FileNotFoundError(f"Model tidak ditemukan: {name}")


def load_whisper(
    model_name: str = "openai/whisper-small",
    language: str = "javanese",
    task: str = "transcribe",
    freeze_encoder: bool = False,
    for_training: bool = False,
) -> Tuple[WhisperProcessor, WhisperForConditionalGeneration]:
    """Muat processor dan model Whisper dengan konfigurasi Bahasa Jawa."""
    path, kwargs = resolve_model(model_name)
    processor = WhisperProcessor.from_pretrained(path, language=language, task=task, **kwargs)
    model = WhisperForConditionalGeneration.from_pretrained(path, **kwargs)

    model.generation_config.language = language
    model.generation_config.task = task
    model.generation_config.forced_decoder_ids = None
    model.generation_config.max_length = None

    if for_training:
        model.config.use_cache = False
    if freeze_encoder:
        model.freeze_encoder()

    return processor, model


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Data collator untuk padding input_features (log-Mel) dan labels Whisper."""

    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch
