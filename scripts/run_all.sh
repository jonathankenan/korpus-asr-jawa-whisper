#!/usr/bin/env bash
# Runner end-to-end pipeline ASR Bahasa Jawa (Whisper)
# Otomatis mendeteksi CUDA GPU vs CPU dan menyesuaikan konfigurasi.
#
# Penggunaan:
#   ./scripts/run_all.sh
#   FOLDS="1 2" MODEL="openai/whisper-base" ./scripts/run_all.sh
#   SKIP_TRAIN=1 ./scripts/run_all.sh

set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python3}"
command -v "$PYTHON_BIN" >/dev/null 2>&1 || PYTHON_BIN="python"
SKIP_INSTALL="${SKIP_INSTALL:-0}"

# 0. Install dependensi Python & Gradio sebelum deteksi CUDA
if [ "$SKIP_INSTALL" = "0" ]; then
  echo "[0/6] Menginstall requirements.txt dan gradio..."
  "$PYTHON_BIN" -m pip install -q -r requirements.txt
  "$PYTHON_BIN" -m pip install -q gradio
fi

# Deteksi CUDA GPU
HAS_CUDA="$("$PYTHON_BIN" -c "import torch; print(1 if torch.cuda.is_available() else 0)" 2>/dev/null || echo 0)"

if [ "$HAS_CUDA" = "1" ]; then
  GPU_NAME="$("$PYTHON_BIN" -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null || echo "CUDA GPU")"
  echo "[INFO] CUDA terdeteksi: $GPU_NAME"
  DEFAULT_MODEL="openai/whisper-small"
  DEFAULT_BEAMS="5"
  DEFAULT_FOLDS="1 2 3 4 5"
  DEFAULT_LIMIT="0"
  DEFAULT_SKIP_TRAIN="0"
else
  echo "[WARN] CUDA tidak tersedia (mode CPU)."
  echo "[WARN] Fine-tuning dilewati otomatis dan baseline memakai konfigurasi ringan."
  echo "[WARN] Set FORCE_CPU_TRAIN=1 jika tetap ingin melatih di CPU."
  DEFAULT_MODEL="openai/whisper-tiny"
  DEFAULT_BEAMS="1"
  DEFAULT_FOLDS="1"
  DEFAULT_LIMIT="50"
  DEFAULT_SKIP_TRAIN="1"
fi

MODEL="${MODEL:-$DEFAULT_MODEL}"
NUM_BEAMS="${NUM_BEAMS:-$DEFAULT_BEAMS}"
FOLDS="${FOLDS:-$DEFAULT_FOLDS}"
LIMIT="${LIMIT:-$DEFAULT_LIMIT}"
SKIP_FETCH="${SKIP_FETCH:-0}"
SKIP_PREPROCESS="${SKIP_PREPROCESS:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-$DEFAULT_SKIP_TRAIN}"
FORCE_CPU_TRAIN="${FORCE_CPU_TRAIN:-0}"
RUN_DEMO="${RUN_DEMO:-1}"

if [ "$FORCE_CPU_TRAIN" = "1" ]; then
  SKIP_TRAIN="0"
fi

echo "=== Konfigurasi Runner ==="
echo "  Python      : $PYTHON_BIN"
echo "  Model       : $MODEL"
echo "  Num beams   : $NUM_BEAMS"
echo "  Folds       : $FOLDS"
echo "  Limit/fold  : $LIMIT (0 = semua)"
echo "  Skip install: $SKIP_INSTALL"
echo "  Skip train  : $SKIP_TRAIN"
echo "  Run demo    : $RUN_DEMO"
echo "=========================="

# 1. Ambil korpus mentah jika belum ada
if [ "$SKIP_FETCH" = "0" ]; then
  if [ ! -d "corpus_k11" ] || [ ! -d "corpus_k05" ]; then
    echo "[1/6] Mengunduh korpus mentah..."
    bash scripts/fetch_data.sh
  else
    echo "[1/6] corpus_k11 & corpus_k05 sudah ada, lewati fetch."
  fi
fi

# 2. Manifest, split, dan normalisasi teks
if [ "$SKIP_PREPROCESS" = "0" ]; then
  echo "[2/6] Membangun manifest, split, dan normalisasi teks..."
  "$PYTHON_BIN" src/build_manifest.py
  "$PYTHON_BIN" src/make_splits.py
  "$PYTHON_BIN" src/normalize_text.py --tokenize

  echo "[2/6] Preprocessing audio (16 kHz, mono, 16-bit)..."
  if [ "$LIMIT" != "0" ]; then
    "$PYTHON_BIN" src/preprocess_audio.py --manifest data/manifest_normalized.csv --out processed_audio --limit "$LIMIT"
  else
    "$PYTHON_BIN" src/preprocess_audio.py --manifest data/manifest_normalized.csv --out processed_audio
  fi
fi

# 3. Baseline zero-shot (pretrained)
echo "[3/6] Inferensi baseline pretrained ($MODEL)..."
for F in $FOLDS; do
  "$PYTHON_BIN" src/infer.py --config configs/inference.yaml --model "$MODEL" --fold "$F" --num-beams "$NUM_BEAMS" --limit "$LIMIT" --tag pretrained
done

# 4. Evaluasi baseline
echo "[4/6] Evaluasi baseline pretrained..."
"$PYTHON_BIN" src/evaluate.py --pred-glob "results/predictions_fold*_pretrained.csv" --tag pretrained

# 5. Fine-tuning + inferensi + evaluasi (khusus CUDA atau FORCE_CPU_TRAIN=1)
if [ "$SKIP_TRAIN" = "0" ]; then
  echo "[5/6] Fine-tuning Whisper..."
  for F in $FOLDS; do
    "$PYTHON_BIN" src/train.py --config configs/train.yaml --model "$MODEL" --fold "$F"
    "$PYTHON_BIN" src/infer.py --config configs/inference.yaml --model "models/whisper-jv-fold${F}" --fold "$F" --num-beams "$NUM_BEAMS" --limit "$LIMIT" --tag finetuned
  done
  "$PYTHON_BIN" src/evaluate.py --pred-glob "results/predictions_fold*_finetuned.csv" --tag finetuned
  FIRST_FOLD="$(echo "$FOLDS" | awk '{print $1}')"
  "$PYTHON_BIN" src/analyze_errors.py --pred "results/predictions_fold${FIRST_FOLD}_finetuned.csv" --tag finetuned
else
  echo "[5/6] Fine-tuning dilewati (mode CPU). Jalankan analisis kesalahan pada hasil baseline..."
  FIRST_FOLD="$(echo "$FOLDS" | awk '{print $1}')"
  "$PYTHON_BIN" src/analyze_errors.py --pred "results/predictions_fold${FIRST_FOLD}_pretrained.csv" --tag pretrained
fi

echo "[6/6] Selesai! Ringkasan hasil tersimpan di folder results/."
if [ "$RUN_DEMO" = "1" ]; then
  DEMO_MODEL="$MODEL"
  if [ "$SKIP_TRAIN" = "0" ] && [ -d "models/whisper-jv-fold${FIRST_FOLD}" ]; then
    DEMO_MODEL="models/whisper-jv-fold${FIRST_FOLD}"
  fi
  echo "[6/6] Menjalankan demo web Gradio dengan model: $DEMO_MODEL"
  echo "      Buka http://127.0.0.1:7860 di browser saat siap (tekan Ctrl+C untuk berhenti)."
  "$PYTHON_BIN" demo/app.py --model "$DEMO_MODEL" --num-beams "$NUM_BEAMS"
else
  echo "      Untuk menjalankan demo web nanti: python demo/app.py --model $MODEL"
  echo "      Lalu buka http://127.0.0.1:7860 di browser saat siap."
fi
