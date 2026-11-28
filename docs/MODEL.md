# Model, Preprocessing, Pelatihan, dan Evaluasi

## 1. Arsitektur dan model pretrained

**Whisper** adalah Transformer encoder-decoder. Encoder menerima log-Mel spectrogram 80 kanal
(16 kHz, jendela 25 ms, geser 10 ms, dipad ke 30 detik), melewati 2 lapisan konvolusi 1D lalu blok
self-attention. Decoder autoregresif memakai self-attention dan cross-attention ke keluaran encoder
untuk memprediksi token BPE. Urutan awal decoder:
`<|startoftranscript|><|jw|><|transcribe|><|notimestamps|>` (`jw` = kode Bahasa Jawa di Whisper).

| Model | Parameter | Revisi (Hugging Face) | Sumber |
|---|---|---|---|
| `openai/whisper-tiny` | 39M | `169d4a4341b33bc18d8881c4b69c2e104e1cc0af` | https://huggingface.co/openai/whisper-tiny |
| `openai/whisper-base` | 74M | `e37978b90ca9030d5170a5c07aadb050351a65bb` | https://huggingface.co/openai/whisper-base |
| `openai/whisper-small` (utama) | 244M | `973afd24965f72e36ca33b3055d56a652f456b4d` | https://huggingface.co/openai/whisper-small |

Lisensi ketiganya Apache-2.0. Revisi dikunci di `configs/models.yaml`; unduh dengan
`python scripts/download_models.py`. Tokenizer dan feature extractor bawaan dipakai tanpa diubah
(tokenizer multibahasa Whisper sudah mencakup alfabet Latin yang dipakai ejaan Jawa di korpus).

## 2. Library

| Library | Versi yang diuji | Dipakai untuk |
|---|---|---|
| PyTorch | 2.12.0 (CPU) | komputasi dan pelatihan |
| Hugging Face Transformers | 5.8.1 (minimum 4.46) | `WhisperProcessor`, `WhisperForConditionalGeneration`, `Seq2SeqTrainer` |
| Accelerate | 1.13.0 | backend `Seq2SeqTrainer` |
| huggingface_hub | bawaan Transformers | unduh bobot pada revisi tertentu |
| NumPy | 2.3.4 | pemrosesan sinyal audio |
| PyYAML | 6.0.3 | konfigurasi |
| Gradio | ≥ 4.0 | demo web (opsional) |

WER/CER dihitung oleh `src/evaluate.py` (Levenshtein sendiri, diuji dengan `--self-test`).

## 3. Preprocessing

Audio (`src/audio.py`, diterapkan di memori, file korpus tidak diubah):
1. Potong segmen menurut `start`/`end` manifest dari file sumber 16 kHz mono 16-bit.
2. Normalisasi puncak ke -3 dBFS (eksperimen `tiny-pretrained-nopeak` mematikannya).
3. Log-Mel spectrogram oleh `WhisperFeatureExtractor`.

Teks (`src/normalize_text.py`): huruf kecil, hapus tag `[...]`, tanda baca, tanda hubung, dan
diakritik. Referensi dan prediksi dinormalisasi dengan fungsi yang sama sebelum dinilai.

## 4. Pelatihan / fine-tuning (`src/train.py`)

| Pengaturan | Default (`configs/train.yaml`) | `tiny-ft` (CPU) | `small-ft` (GPU) |
|---|---|---|---|
| Bagian yang dilatih | seluruh model | decoder saja (encoder beku) | seluruh model |
| Epoch | 5 | 2 | 5 |
| Batch × akumulasi gradien | 8 × 2 | 8 × 1 | 8 × 2 |
| Learning rate | 1e-5 | dipilih dari {1e-5, 5e-5} | dipilih dari {5e-6, 1e-5, 3e-5} |
| Warmup (step) | 100 | 50 | 100 |
| Optimizer | AdamW, weight decay 0.01, jadwal linear | sama | sama |
| Mixed precision | fp16 bila ada GPU | tidak | fp16 |
| Seed | 42 | 42 | 42 |

- Train/validation memakai segmen berdurasi 0,3–30 s; test dievaluasi utuh.
- Label = `text_norm`. Padding label diabaikan pada loss (-100).
- Setiap epoch, WER validation dihitung dengan generasi greedy (`no_repeat_ngram_size=3`, sama
  dengan inferensi); epoch dengan WER terkecil disimpan sebagai model akhir.
- Learning rate dipilih dari WER validation fold 1, lalu dipakai untuk kelima fold. Catatan:
  penutur validation fold 1 adalah penutur test fold 2, sehingga ada sedikit kebocoran pemilihan
  hyperparameter pada fold 2.

## 5. Decoding dan inferensi (`src/infer.py`, `configs/inference.yaml`)

Greedy (`num_beams=1`), `max_new_tokens=225`, `no_repeat_ngram_size=3`, tanpa timestamp, batch 8.
Variasi beam search 5 diuji pada `tiny-pretrained-beam5`. Audio lebih dari 30 s dipotong berurutan
(hanya satu segmen pada korpus ini).

## 6. Evaluasi (`src/evaluate.py`, `src/analyze_errors.py`)

- WER dan CER agregat (total kesalahan / total kata atau karakter referensi), per korpus, per fold
  (rata-rata ± simpangan baku), per penutur, dialek, status penutur, gender, durasi, kejadian, bahasa.
- **Subset bersih** sebagai pembanding: tanpa pertanyaan panduan, segmen bertanda hening/halusinasi,
  segmen berlabel bukan Jawa, referensi ber-tag atau berangka, durasi < 1 s, atau > 25 karakter/s.
- **Keluarga dialek** (tingkat penutur): Timuran, Semarangan, Solo-Jogja (Mataraman), Campuran,
  Tidak dirinci. Aturan pemetaan ada di `src/data.py`.

## 7. Eksperimen

Definisi lengkap di `configs/experiments.yaml`, hasil di `results/summary.md`.

## 8. Tautan model hasil pelatihan

Belum ada tautan. Model lokal ada di `models/<eksperimen>/fold<k>/` (tidak di-commit).
Isi tautan penyimpanan setelah model diunggah.

## 9. Referensi

- Radford, A., Kim, J. W., Xu, T., Brockman, G., McLeavey, C., & Sutskever, I. (2022).
  *Robust Speech Recognition via Large-Scale Weak Supervision*. arXiv:2212.04356.
- Hugging Face, dokumentasi Whisper: https://huggingface.co/docs/transformers/model_doc/whisper
- Gandhi, S. (2022). *Fine-Tune Whisper For Multilingual ASR with Transformers*.
  https://huggingface.co/blog/fine-tune-whisper — `DataCollatorSpeechSeq2SeqWithPadding`
  (`src/model.py`) dan pola `Seq2SeqTrainer` (`src/train.py`) diadaptasi dari sini.
- Repositori model: https://github.com/openai/whisper
