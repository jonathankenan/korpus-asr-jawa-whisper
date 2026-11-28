# ASR Bahasa Jawa dengan Whisper

Tugas Besar Fase 2 IF4071 Pemrosesan Ucapan (Semester 1 2026/2027, STEI ITB): membangun sistem pengenal ucapan (ASR) Bahasa Jawa dengan teknik Whisper, memakai korpus Fase 1 kelompok ini yang digabung dengan korpus kelompok 5.

Status: **dalam pengerjaan**. Hasil eksperimen yang sudah selesai ada di `results/summary.md`.

## Anggota kelompok

Danendra Shafi Athallah (13523136), Jonathan Kenan Budianto (13523139), Jovandra Otniel Panagalambok Siregar (13523141), Muhammad Farrel Wibowo (13523153), Lukas Raja Agripa (13523158).

## Sumber data

| Singkatan | Korpus | Repo |
|---|---|---|
| `corpus_k11` | Korpus Bahasa Jawa kelompok ini (Fase 1) | https://github.com/rlukassa/Korpus_Bahasa_Jawa |
| `corpus_k05` | Korpus Bahasa Jawa kelompok 5 | https://github.com/angelinaefrina/Korpus_Bahasa_Jawa |

Penutur kelompok 5 (S01 sampai S10 pada repo asalnya) dinomori ulang menjadi S11 sampai S20 pada data gabungan (`data/id_mapping.csv`).

Audio **tidak di-commit** ke repo ini. Rincian dataset, pembagian data, dan keterbatasannya ada di `data/DATASET.md`.

## 1. Persiapan

Butuh Python 3.10 atau lebih baru.

```bash
git clone https://github.com/jonathankenan/korpus-asr-jawa-whisper.git
cd korpus-asr-jawa-whisper
pip install -r requirements.txt
./scripts/fetch_data.sh                 # clone kedua korpus pada commit yang dikunci (±370 MB dan ±570 MB)
python scripts/download_models.py       # bobot whisper tiny, base, small (±1,4 GB)
```

Jika korpus sudah ada di folder lain (pada commit yang sama), lewati `fetch_data.sh` dan arahkan lewat variabel lingkungan (PowerShell):

```powershell
$env:CORPUS_K11 = "..\Korpus_Bahasa_Jawa_01"
$env:CORPUS_K05 = "..\Korpus_Bahasa_Jawa"
```

## 2. Menjalankan semua eksperimen

```bash
./scripts/run_all.sh                                  # GPU -> grup gpu; tanpa GPU -> grup local
python src/run_experiments.py --list                  # daftar eksperimen
python src/run_experiments.py tiny-pretrained         # satu eksperimen
python src/run_experiments.py --group local --keep-awake
```

Proses dapat dilanjutkan: fold yang sudah dilatih dan segmen yang sudah ditranskripsi dilewati.

**Fine-tuning whisper-small di GPU:** buka `notebooks/finetune_gpu.ipynb` di Google Colab atau Kaggle dan ikuti petunjuk di dalamnya.

## 3. Menjalankan tiap tahap

| Tahap | Perintah | Keluaran |
|---|---|---|
| Manifest gabungan | `python src/build_manifest.py` | `data/manifest.csv` |
| Split antar penutur | `python src/make_splits.py` | `data/splits/` |
| Normalisasi teks | `python src/normalize_text.py --tokenize` | `data/manifest_normalized.csv` |
| Preprocessing audio ke file (opsional) | `python src/preprocess_audio.py` | `processed_audio/` |
| Fine-tuning satu fold | `python src/train.py --exp nama --model openai/whisper-small --fold 1` | `models/nama/fold1/`, `results/training/nama/` |
| Inferensi test | `python src/infer.py --exp nama --model models/nama/fold1 --fold 1` | `results/predictions/nama/` |
| Inferensi pretrained 5 fold | `python src/infer.py --exp nama --model openai/whisper-small --fold all` | `results/predictions/nama/` |
| Evaluasi WER/CER | `python src/evaluate.py` | `results/summary.md`, `results/eval/` |
| Analisis kesalahan | `python src/analyze_errors.py --exp nama --baseline pembanding` | `results/analysis/nama.md` |

Training dan inferensi memotong segmen langsung dari audio sumber dan menerapkan preprocessing di memori, jadi `preprocess_audio.py` tidak wajib dijalankan.

Uji mandiri metrik: `python src/evaluate.py --self-test` dan `python src/analyze_errors.py --self-test`.

## 4. Demo

```bash
python demo/app.py --model openai/whisper-small                    # web di http://127.0.0.1:7860
python demo/app.py --audio contoh.wav --model models/small-ft/fold1  # satu file, tanpa web
```

## 5. Eksperimen

| Eksperimen | Isi | Variasi yang diuji | Perangkat |
|---|---|---|---|
| `tiny-pretrained`, `base-pretrained`, `small-pretrained` | tanpa fine-tuning | ukuran model | CPU |
| `tiny-pretrained-nopeak` | tanpa normalisasi amplitudo | preprocessing | CPU |
| `tiny-pretrained-beam5` | beam search 5 | decoding | CPU |
| `tiny-ft-lr1e-5`, `tiny-ft-lr5e-5`, `tiny-ft` | fine-tune decoder tiny | learning rate; pretrained vs fine-tuned | CPU |
| `small-ft-lr*`, `small-ft` | fine-tune penuh small | learning rate; pretrained vs fine-tuned | GPU |

Konfigurasi: `configs/experiments.yaml`, `configs/train.yaml`, `configs/inference.yaml`. Penjelasan model dan metode: `docs/MODEL.md`.

## Struktur repo

```
src/        manifest, split, preprocessing, training, inferensi, evaluasi, analisis, runner
data/       manifest gabungan, split antar penutur, dokumentasi dataset
configs/    konfigurasi model, training, inferensi, eksperimen
scripts/    fetch_data.sh, download_models.py, run_all.sh
notebooks/  fine-tuning di GPU
results/    prediksi, metrik, analisis kesalahan, ringkasan
report/     bahan laporan
demo/       demo transkripsi
docs/       dokumentasi model dan metode
corpus_k11/ (lokal, tidak di-commit)
corpus_k05/ (lokal, tidak di-commit)
models/     (lokal, tidak di-commit; tautan model dicatat di docs/MODEL.md)
```

## Catatan

- Transkripsi korpus kelompok ini masih **draf ASR dan belum dikoreksi manusia**. Skor pada korpus ini tidak dapat dianggap akurasi sebenarnya; hasil dilaporkan terpisah per korpus.
- Setelah transkripsi dikoreksi: perbarui commit di `scripts/fetch_data.sh`, jalankan `build_manifest.py` dan `normalize_text.py`, lalu `evaluate.py` (skor dihitung ulang tanpa inferensi ulang). Fine-tuning perlu diulang agar memakai teks baru. Jangan menghapus atau mengubah urutan baris anotasi karena `utt_id` mengikuti urutan baris.
- Pembagian data memakai split antar penutur: penutur pada test set tidak muncul di train atau validation pada fold yang sama.
- File di `results/analysis/` dan contoh prediksi memuat kutipan transkripsi kedua korpus. Periksa ketentuan penyebaran tiap korpus sebelum mempublikasikannya.
