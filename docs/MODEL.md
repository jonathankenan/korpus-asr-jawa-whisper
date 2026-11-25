# Implementasi Model & Pelatihan Whisper

## 1. Arsitektur dan Model Pretrained

Sistem ASR menggunakan arsitektur **Whisper** (Transformer encoder-decoder):
- **Encoder**: 2 lapisan konvolusi 1D + blok Transformer encoder yang memproses log-Mel spectrogram 80-channel (16 kHz, window 25 ms, stride 10 ms).
- **Decoder**: blok Transformer decoder autoregresif dengan cross-attention ke representasi encoder untuk memprediksi urutan token BPE teks target (`language="javanese"`, `task="transcribe"`).
- **Varian model**: `openai/whisper-tiny` (39M), `openai/whisper-base` (74M), `openai/whisper-small` (244M, utama), `openai/whisper-medium` (769M).

## 2. Library & Referensi

- **PyTorch** (`torch>=2.1.0`) — backend komputasi tensor dan pelatihan.
- **Hugging Face Transformers** (`transformers>=4.37.0`) — `WhisperProcessor`, `WhisperForConditionalGeneration`, `Seq2SeqTrainer`.
- **Referensi utama**:
  - Radford, A. et al. (2022). *Robust Speech Recognition via Large-Scale Weak Supervision*. OpenAI.
  - Dokumentasi Whisper Hugging Face: https://huggingface.co/docs/transformers/model_doc/whisper

## 3. Cara Menjalankan Pelatihan

Verifikasi data & fold (tanpa GPU):
```bash
python src/train.py --config configs/train.yaml --fold 1 --dry-run
```

Fine-tuning penuh pada Fold 1:
```bash
python src/train.py --config configs/train.yaml --fold 1
```

Menjalankan variasi eksperimen (learning rate / ukuran model):
```bash
python src/train.py --fold 1 --model openai/whisper-base --lr 3e-5 --epochs 5
```

Model terbaik tiap fold disimpan ke `models/whisper-jv-fold<k>/` (tidak di-commit karena ukuran besar; tautan model dicatat di sini setelah pelatihan di GPU).
