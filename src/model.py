"""Utilitas model Whisper, data collator, dan pemuat audio untuk ASR Bahasa Jawa.

Referensi:
  - Radford et al. (2022), "Robust Speech Recognition via Large-Scale Weak Supervision"
  - Hugging Face Transformers: WhisperForConditionalGeneration & WhisperProcessor
    https://huggingface.co/docs/transformers/model_doc/whisper
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import struct
import wave

import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor


def load_whisper(
    model_name: str = "openai/whisper-small",
    language: str = "javanese",
    task: str = "transcribe",
    freeze_feature_encoder: bool = False,
) -> Tuple[WhisperProcessor, WhisperForConditionalGeneration]:
    """Muat processor dan model Whisper pretrained dengan konfigurasi Bahasa Jawa."""
    processor = WhisperProcessor.from_pretrained(model_name, language=language, task=task)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)

    model.generation_config.language = language
    model.generation_config.task = task
    model.generation_config.forced_decoder_ids = None
    model.config.use_cache = False

    if freeze_feature_encoder:
        model.freeze_feature_encoder()

    return processor, model


def load_wav_segment(
    audio_path: Path,
    start_sec: Optional[float] = None,
    end_sec: Optional[float] = None,
    target_sr: int = 16_000,
) -> np.ndarray:
    """Baca file WAV (atau potong segmen) menjadi array float32 16 kHz mono."""
    with wave.open(str(audio_path), "rb") as wf:
        sr = wf.getframerate()
        nc = wf.getnchannels()
        sw = wf.getsampwidth()
        n_frames = wf.getnframes()

        start_frame = 0
        end_frame = n_frames
        if start_sec is not None and end_sec is not None:
            start_frame = max(0, int(start_sec * sr))
            end_frame = min(n_frames, int(end_sec * sr))

        wf.setpos(start_frame)
        raw = wf.readframes(max(0, end_frame - start_frame))

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
