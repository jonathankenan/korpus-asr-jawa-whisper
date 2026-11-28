"""Demo sistem ASR Bahasa Jawa dengan Whisper (CLI & web Gradio), spesifikasi Fase 2 §8.

Memakai pemuat model, preprocessing, dan decoding yang sama dengan evaluasi
(src/ dan configs/inference.yaml).

Cara menjalankan:
  1. Web (butuh gradio):      python demo/app.py --model models/small-ft/fold1
  2. Satu file (CLI):         python demo/app.py --audio path/ke/audio.wav --model openai/whisper-small
  3. Cek skrip tanpa model:   python demo/app.py --dry-run
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
import yaml  # noqa: E402

from audio import load_wav, peak_normalize  # noqa: E402
from infer import transcribe_many  # noqa: E402
from model import load_whisper  # noqa: E402
from normalize_text import normalize  # noqa: E402


class WhisperJavaneseDemo:
    """Transkripsi file WAV Bahasa Jawa dengan model pretrained atau hasil fine-tuning."""

    def __init__(self, model_id: str, num_beams: int = 0):
        with open(ROOT / "configs" / "inference.yaml", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        self.audio_cfg = cfg["audio"]
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor, self.model = load_whisper(model_id, cfg["model"]["language"], cfg["model"]["task"])
        self.model.to(self.device).eval()
        self.gen_kwargs = {
            "language": cfg["model"]["language"],
            "task": cfg["model"]["task"],
            "num_beams": num_beams or cfg["decoding"]["num_beams"],
            "do_sample": False,
            "max_new_tokens": cfg["decoding"]["max_new_tokens"],
            "no_repeat_ngram_size": cfg["decoding"]["no_repeat_ngram_size"],
        }

    def transcribe_file(self, wav_path: Path) -> Tuple[str, str]:
        sr = self.audio_cfg["sample_rate"]
        audio = load_wav(wav_path, target_sr=sr)
        if self.audio_cfg["peak_dbfs"] is not None:
            audio = peak_normalize(audio, self.audio_cfg["peak_dbfs"])
        raw = transcribe_many([audio], self.processor, self.model, self.device, self.gen_kwargs,
                              sr, self.audio_cfg["chunk_length_sec"])[0]
        return raw, normalize(raw)


def launch_gradio(engine: WhisperJavaneseDemo, port: int = 7860) -> None:
    import gradio as gr

    def predict(audio_filepath: str):
        if not audio_filepath:
            return "", ""
        return engine.transcribe_file(Path(audio_filepath))

    ui = gr.Interface(
        fn=predict,
        inputs=gr.Audio(sources=["upload", "microphone"], type="filepath", format="wav",
                        label="Audio Bahasa Jawa (WAV)"),
        outputs=[
            gr.Textbox(label="Transkripsi Model (Mentah)"),
            gr.Textbox(label="Transkripsi Ternormalisasi"),
        ],
        title="Demo ASR Bahasa Jawa (Whisper)",
        description=f"Model aktif: `{engine.model_id}` | Device: `{engine.device}`",
    )
    print(f"Demo web siap! Buka http://127.0.0.1:{port} di browser Anda.")
    ui.launch(server_name="127.0.0.1", server_port=port)


def main():
    parser = argparse.ArgumentParser(description="Demo ASR Bahasa Jawa dengan Whisper")
    parser.add_argument("--model", default="openai/whisper-small", help="Nama atau folder model Whisper")
    parser.add_argument("--audio", default=None, help="Path file WAV untuk mode CLI")
    parser.add_argument("--num-beams", type=int, default=0, help="0 = ikut configs/inference.yaml")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--dry-run", action="store_true", help="Verifikasi skrip demo tanpa memuat model")
    args = parser.parse_args()

    if args.dry_run:
        print(f"Dry-run demo OK | Model target: {args.model} | "
              f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
        return

    engine = WhisperJavaneseDemo(model_id=args.model, num_beams=args.num_beams)

    if args.audio:
        wav_path = Path(args.audio)
        if not wav_path.is_absolute() and not wav_path.exists():
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
            path_str = input("Path WAV > ").strip().strip('"')
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
