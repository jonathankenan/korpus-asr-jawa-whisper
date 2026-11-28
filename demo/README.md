# Demo Sistem ASR Bahasa Jawa

Folder ini berisi aplikasi demo interaktif untuk mendemonstrasikan transkripsi audio Bahasa Jawa menggunakan model Whisper (pretrained atau hasil fine-tuning).

## Cara Menjalankan

### 1. Mode Web UI (Gradio)
```bash
pip install gradio
python demo/app.py --model openai/whisper-small
```
Atau arahkan ke model hasil fine-tuning:
```bash
python demo/app.py --model models/small-ft/fold1
```

Demo memakai preprocessing dan decoding yang sama dengan evaluasi (`configs/inference.yaml`). Masukan harus berupa file WAV.

### 2. Mode CLI (Tanpa Gradio)
```bash
python demo/app.py --audio path/to/sample.wav --model openai/whisper-small
```

Jika dijalankan tanpa `--audio` dan `gradio` belum terpasang, skrip otomatis masuk ke mode CLI interaktif.
