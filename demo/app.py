"""Demo sistem ASR Bahasa Jawa dengan Whisper (CLI & Gradio Web UI).

Sesuai SPEC.md bagian 8 (Demonstrasi sistem):
  Memasukkan audio Bahasa Jawa (.wav) dan menampilkan hasil transkripsi model.

Cara menjalankan:
  1. Mode Web UI (jika gradio terpasang):
       python demo/app.py --model openai/whisper-small
  2. Mode CLI satu file:
       python demo/app.py --audio path/to/audio.wav --model openai/whisper-small
  3. Verifikasi tanpa memuat bobot model:
       python demo/app.py --dry-run
"""

import argparse
import re
import struct
import unicodedata
import wave
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

ROOT = Path(__file__).resolve().parent.parent
TARGET_SR = 16_000


def normalize_output(text: str) -> str:
    """Normalisasi ringan untuk tampilan hasil transkripsi."""
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = re.sub(r"\[[^\]]*\]", " ", text).lower()
    text = re.sub(r"[^a-z0-9'\- ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_wav_16k(path: Path) -> np.ndarray:
    """Baca file WAV menjadi array float32 mono 16 kHz."""
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        nc = wf.getnchannels()
        sw = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())

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

    if sr != TARGET_SR and len(arr) > 0:
        n_out = int(round(len(arr) * TARGET_SR / sr))
        x_old = np.linspace(0.0, 1.0, num=len(arr), endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
        arr = np.interp(x_new, x_old, arr).astype(np.float32)

    return arr


class WhisperJavaneseDemo:
    """Wrapper demo untuk transkripsi audio Bahasa Jawa dengan Whisper."""

    def __init__(self, model_id: str = "openai/whisper-small", num_beams: int = 5):
        self.model_id = model_id
        self.num_beams = num_beams
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = WhisperProcessor.from_pretrained(
            model_id, language="javanese", task="transcribe"
        )
        self.model = WhisperForConditionalGeneration.from_pretrained(model_id).to(self.device)
        self.model.eval()
        self.model.generation_config.language = "javanese"
        self.model.generation_config.task = "transcribe"
        self.model.generation_config.forced_decoder_ids = None
        self.model.generation_config.max_length = None

    def transcribe_file(self, wav_path: Path) -> Tuple[str, str]:
        audio = load_wav_16k(wav_path)
        chunk_samples = 30 * TARGET_SR
        parts = []
        for start in range(0, len(audio), chunk_samples):
            chunk = audio[start : start + chunk_samples]
            inputs = self.processor(chunk, sampling_rate=TARGET_SR, return_tensors="pt")
            with torch.no_grad():
                ids = self.model.generate(
                    inputs.input_features.to(self.device),
                    num_beams=self.num_beams,
                    max_new_tokens=225,
                )
            text = self.processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
            if text:
                parts.append(text)
        raw = " ".join(parts).strip()
        return raw, normalize_output(raw)


def launch_gradio(demo_engine: WhisperJavaneseDemo, port: int = 7860) -> None:
    import gradio as gr

    def predict(audio_filepath: str):
        if not audio_filepath:
            return "", ""
        return demo_engine.transcribe_file(Path(audio_filepath))

    ui = gr.Interface(
        fn=predict,
        inputs=gr.Audio(sources=["upload", "microphone"], type="filepath", label="Audio Bahasa Jawa (WAV)"),
        outputs=[
            gr.Textbox(label="Transkripsi Model (Mentah)"),
            gr.Textbox(label="Transkripsi Ternormalisasi"),
        ],
        title="Demo ASR Bahasa Jawa (Whisper)",
        description=f"Model aktif: `{demo_engine.model_id}` | Device: `{demo_engine.device}`",
    )
    ui.launch(server_port=port)


def main():
    parser = argparse.ArgumentParser(description="Demo ASR Bahasa Jawa dengan Whisper")
    parser.add_argument("--model", default="openai/whisper-small", help="Nama atau path model Whisper")
    parser.add_argument("--audio", default=None, help="Path file WAV untuk mode CLI")
    parser.add_argument("--num-beams", type=int, default=5)
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--dry-run", action="store_true", help="Verifikasi skrip demo tanpa memuat model")
    args = parser.parse_args()

    if args.dry_run:
        print(f"Dry-run demo OK | Model target: {args.model} | Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
        return

    engine = WhisperJavaneseDemo(model_id=args.model, num_beams=args.num_beams)

    if args.audio:
        wav_path = Path(args.audio)
        if not wav_path.is_absolute():
            wav_path = ROOT / wav_path
        raw, norm = engine.transcribe_file(wav_path)
        print(f"File       : {wav_path}")
        print(f"Transkripsi: {raw}")
        print(f"Normalisasi: {norm}")
        return

    try:
        launch_gradio(engine, port=args.port)
    except ImportError:
        print("Gradio tidak terpasang. Jalankan mode CLI interaktif (masukkan path .wav, atau 'q' untuk keluar):")
        while True:
            path_str = input("Path WAV > ").strip()
            if not path_str or path_str.lower() in ("q", "quit", "exit"):
                break
            p = Path(path_str)
            if not p.exists():
                print(f"File tidak ditemukan: {p}")
                continue
            raw, norm = engine.transcribe_file(p)
            print(f"  Transkripsi: {raw}")
            print(f"  Normalisasi: {norm}\n")


if __name__ == "__main__":
    main()
