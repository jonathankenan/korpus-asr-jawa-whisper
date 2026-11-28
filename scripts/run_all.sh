#!/usr/bin/env bash
# Runner end-to-end pipeline ASR Bahasa Jawa (Whisper).
#
#   ./scripts/run_all.sh                 # GPU -> eksperimen grup gpu, tanpa GPU -> grup local
#   GROUP=local ./scripts/run_all.sh
#   SKIP_INSTALL=1 ./scripts/run_all.sh
#
# Jika korpus sudah ada di folder lain: set CORPUS_K11 dan CORPUS_K05 (fetch dilewati).
# Semua langkah dapat dilanjutkan: ulangi perintah yang sama jika terputus.
# Skenario eksperimen ada di configs/experiments.yaml.

set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python}"
command -v "$PYTHON_BIN" >/dev/null 2>&1 || PYTHON_BIN="python3"
export PYTHONUTF8=1
SKIP_INSTALL="${SKIP_INSTALL:-0}"

if [ "$SKIP_INSTALL" = "0" ]; then
  echo "[1/4] Memasang dependensi..."
  "$PYTHON_BIN" -m pip install -q -r requirements.txt
fi

HAS_CUDA="$("$PYTHON_BIN" -c "import torch; print(1 if torch.cuda.is_available() else 0)" 2>/dev/null || echo 0)"
if [ "$HAS_CUDA" = "1" ]; then
  GROUP="${GROUP:-gpu}"
else
  GROUP="${GROUP:-local}"
fi

if [ -z "${CORPUS_K11:-}" ] && [ -z "${CORPUS_K05:-}" ]; then
  if [ ! -d corpus_k11 ] || [ ! -d corpus_k05 ]; then
    echo "[2/4] Mengunduh korpus..."
    bash scripts/fetch_data.sh
  else
    echo "[2/4] corpus_k11 dan corpus_k05 sudah ada."
  fi
else
  echo "[2/4] Memakai korpus dari CORPUS_K11 / CORPUS_K05."
fi

echo "[3/4] Mengunduh bobot model (dilewati jika sudah ada di cache)..."
if [ "$GROUP" = "gpu" ]; then
  "$PYTHON_BIN" scripts/download_models.py openai/whisper-small
else
  "$PYTHON_BIN" scripts/download_models.py
fi

echo "[4/4] Menjalankan eksperimen grup '$GROUP'..."
if [ "$GROUP" = "gpu" ]; then
  # fine-tuning small dibandingkan dengan baseline small
  "$PYTHON_BIN" src/run_experiments.py small-pretrained
fi
"$PYTHON_BIN" src/run_experiments.py --group "$GROUP"

echo "Selesai. Ringkasan: results/summary.md | analisis kesalahan: results/analysis/"
echo "Demo web: python demo/app.py --model <nama atau folder model>"
