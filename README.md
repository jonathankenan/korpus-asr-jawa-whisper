# ASR Bahasa Jawa dengan Whisper

Tugas Besar Fase 2 IF4071 Pemrosesan Ucapan (Semester 1 2026/2027, STEI ITB): membangun sistem pengenal ucapan (ASR) Bahasa Jawa dengan teknik Whisper, memakai korpus Fase 1 kelompok ini yang digabung dengan korpus kelompok 5.

Status: **dalam pengerjaan**. Belum ada hasil eksperimen.

## Anggota kelompok

Danendra Shafi Athallah (13523136), Jonathan Kenan Budianto (13523139), Jovandra Otniel Panagalambok Siregar (13523141), Muhammad Farrel Wibowo (13523153), Lukas Raja Agripa (13523158).

## Sumber data

| Singkatan | Korpus | Repo |
|---|---|---|
| `corpus_k11` | Korpus Bahasa Jawa kelompok ini (Fase 1) | https://github.com/rlukassa/Korpus_Bahasa_Jawa |
| `corpus_k05` | Korpus Bahasa Jawa kelompok 5 | https://github.com/angelinaefrina/Korpus_Bahasa_Jawa |

Penutur kelompok 5 (S01 sampai S10 pada repo asalnya) dinomori ulang menjadi S11 sampai S20 pada data gabungan. Pemetaannya akan dicatat di `data/id_mapping.csv`.

Audio dan data mentah **tidak di-commit** ke repo ini. Kedua korpus diambil ke folder lokal yang di-gitignore.
## Cara menyiapkan data

```bash
git clone https://github.com/jonathankenan/korpus-asr-jawa-whisper.git
cd korpus-asr-jawa-whisper
./scripts/fetch_data.sh
```

Skrip meng-clone kedua korpus ke `corpus_k11/` dan `corpus_k05/` lalu mengunci keduanya pada versi tertentu agar semua anggota memakai data yang sama. Ukurannya sekitar 370 MB dan 570 MB.

Setelah data terambil, buat manifest gabungan dan split:

```bash
python3 src/build_manifest.py
python3 src/make_splits.py
```

Rincian dataset, strategi split, dan keterbatasannya ada di `data/DATASET.md`.

## Struktur repo

```
src/        kode: manifest, preprocessing, training, inferensi, evaluasi
data/       manifest gabungan, split antar penutur, pedoman transkripsi
configs/    konfigurasi eksperimen
scripts/    skrip bantu (fetch_data.sh)
results/    metrik, contoh prediksi, analisis kesalahan
report/     laporan dan slide
demo/       demo transkripsi
docs/       catatan dan dokumentasi tambahan
corpus_k11/ (lokal, tidak di-commit)
corpus_k05/ (lokal, tidak di-commit)
models/     (lokal, tidak di-commit; link model dicatat di docs/)
```

## Catatan

- Transkripsi korpus kelompok ini masih draf ASR dan wajib dikoreksi manusia sebelum dipakai untuk pelatihan atau evaluasi.
- Pembagian data memakai split antar penutur: penutur pada test set tidak boleh muncul di train.
