# Dokumentasi dataset gabungan

Dataset ini menyatukan korpus Bahasa Jawa kelompok ini (`corpus_k11`, ID S01 sampai S10) dan korpus kelompok 5 (`corpus_k05`, ID asal S01 sampai S10, dinomori ulang menjadi S11 sampai S20). Sumber korpus tercantum di README, dan versi yang dipakai dikunci di `scripts/fetch_data.sh`. Semua angka di bawah dihitung oleh `src/build_manifest.py` dan `src/make_splits.py`.

Korpus kelompok 5 dipakai pada rilis `v1.1-koreksi` (commit `20047ca`), yang memperbaiki nama berkas S06, stempel waktu S03 dan S08, segmen tumpang tindih, dan transkripsi yang tertukar pada anotasi mereka (rincian di `CHANGES.md` pada repo mereka).

## Ringkasan

| | Kelompok ini (k11) | Kelompok 5 (k05) | Gabungan |
|---|---|---|---|
| Penutur | 10 (5 L, 5 P) | 10 (5 L, 5 P) | 20 (10 L, 10 P) |
| Segmen anotasi | 2.188 | 922 | 3.110 |
| Durasi audio | 108,6 menit | 115,5 menit | 224,1 menit |
| Durasi segmen bertranskripsi | 95,6 menit | 100,3 menit | 196,0 menit |
| File audio | 10 (1 per penutur) | 40 (4 per penutur) | 50 |

Durasi bertranskripsi adalah jumlah durasi segmen yang berteks dan berdurasi positif. Angka ini belum disaring oleh aturan apa pun (lihat bagian Tanda pada baris).

## Penutur

| ID baru | Asal | Gender | Dialek | Status penutur |
|---|---|---|---|---|
| S01 | k11 | L | Jawa Timuran (Suroboyoan) | native |
| S02 | k11 | L | Jawa Tengahan (Semarangan) | native |
| S03 | k11 | L | Jawa Tengahan (Solo) | non-native |
| S04 | k11 | L | Jawa Mataraman (Jogja) | native |
| S05 | k11 | L | Jawa Tengahan/Mataraman (ngoko) | heritage |
| S06 | k11 | P | Jawa Tengahan (Semarangan) | native |
| S07 | k11 | P | Jawa Timuran (Suroboyoan) | native |
| S08 | k11 | P | Jawa Timuran (Suroboyoan) | native |
| S09 | k11 | P | Jawa campuran (Tengahan + Malangan) | non-native |
| S10 | k11 | P | Jawa Timuran (Trenggalekan/Mataraman wetan) | native |
| S11 | k05 (S01) | L | Jawa | non-native |
| S12 | k05 (S02) | L | Jawa | non-native |
| S13 | k05 (S03) | P | Jawa Solo | native |
| S14 | k05 (S04) | L | Jawa Timuran | native |
| S15 | k05 (S05) | L | Jawa Jogja | native |
| S16 | k05 (S06) | L | Jawa Jogja | native |
| S17 | k05 (S07) | P | Jawa Jogja | native |
| S18 | k05 (S08) | P | Jawa Jogja | native |
| S19 | k05 (S09) | P | Jawa Jogja | native |
| S20 | k05 (S10) | P | Jawa Jogja | native |

Sumber label: untuk k11, dialek dan status dibaca dari kolom `label_dialek` pada anotasi (penutur tanpa label non-asli atau warisan dicatat native). Untuk k05, dari kolom `Language_Dialect` dan `Speaker_Type` pada `metadata.csv` mereka. Tabel lengkap ada di `data/speakers.csv`, pemetaan ID di `data/id_mapping.csv`.

## Manifest (`data/manifest.csv`)

Satu baris per segmen anotasi, semua baris dipertahankan. Kolom utama: `utt_id`, `speaker`, `group`, `orig_speaker`, `orig_file`, `audio_path` (relatif terhadap akar repo), `start`, `end`, `duration`, `text_raw`, `language_label`, `language_label_src`, `dialect`, `topic`, `event`, `asr_confidence`, `gender`, `speaker_status`, `transcript_status`, `flags`, `usable`.

- `text_raw` adalah teks sumber tanpa perubahan. Belum ada normalisasi.
- `transcript_status`: `draft_asr` untuk k11 (transkripsi hasil ASR yang belum dikoreksi manusia) dan `unverified` untuk k05 (status review tidak terdokumentasi).
- `language_label` hanya tersedia untuk k11. Nilainya sudah dikoreksi pada 14 segmen yang berlabel Inggris padahal teksnya tidak memuat kata Inggris (daftar koreksi di `LABEL_FIXES` pada `src/build_manifest.py`). `language_label_src` menyimpan label asal dari anotasi.
- Kolom `label_speaker` pada anotasi k11 tidak disalin karena memuat nama penutur.

### Tanda pada baris (`flags`)

Tanda hanya menandai baris. Tidak ada baris yang dibuang oleh skrip.

| Tanda | Arti | k11 | k05 |
|---|---|---|---|
| `no_text` | tidak ada transkripsi (baris `-gap`) | 20 | 0 |
| `short_lt1s` | durasi di bawah 1 detik | 252 | 6 |
| `long_gt30s` | durasi di atas 30 detik | 1 | 0 |
| `neg_duration` | waktu selesai tidak lebih besar dari waktu mulai | 0 | 0 |
| `beyond_audio` | waktu selesai melewati akhir audio | 0 | 0 |
| `filename_fixed` | nama berkas pada anotasi tidak cocok, dicocokkan dengan mengganti `_` menjadi `-` | 0 | 0 |

### Kolom `usable`

`usable` bernilai `no` untuk 20 baris yang tidak mungkin dipakai: yang bertanda `no_text`, `neg_duration`, atau `beyond_audio` (atau `audio_missing`, yang tidak terjadi pada data ini). Sisanya, 3.090 baris, bernilai `yes`. Tanda `short_lt1s`, `long_gt30s`, dan `filename_fixed` tidak memengaruhi `usable`.

### Label bahasa (k11, segmen berteks, setelah koreksi)

| Label | Segmen | Menit |
|---|---|---|
| Jawa | 1.870 | 77,1 |
| Indonesia | 176 | 5,1 |
| Campur (Jawa-Indonesia) | 86 | 10,7 |
| Campur (Jawa-Inggris) | 22 | 1,6 |
| Campur (Jawa-Indonesia-Inggris) | 5 | 0,9 |
| Inggris | 9 | 0,3 |

Koreksi hanya mengubah label yang menyebut Inggris tanpa ada kata Inggris di teksnya. Pada 11 segmen, satu-satunya kata Inggris adalah "So" atau "is" (misalnya `S04_0128`, `S08_0061`), kemungkinan salah dengar ASR, dan labelnya dibiarkan. Label "Inggris" murni pada beberapa segmen yang isinya campuran juga tidak diubah (misalnya `S10_0020`).

## Strategi pembagian data

Pembagian dilakukan antar penutur (speaker-independent) sesuai spesifikasi Fase 2: penutur pada test set tidak muncul di train atau validation pada fold yang sama.

Dengan 20 penutur dipakai **5-fold cross-validation antar penutur**. Tiap penutur menjadi test tepat satu kali. Pada tiap fold:

- **Test:** 4 penutur (2 laki-laki dan 2 perempuan, 2 dari k11 dan 2 dari k05).
- **Validation:** 2 penutur (1 laki-laki dari k11 dan 1 perempuan dari k05), diambil dari penutur test fold berikutnya secara rotasi.
- **Train:** 14 penutur sisanya.

Penutur non-native dan heritage (S03, S05, S09, S11, S12) disebar agar tidak menumpuk pada satu fold. Skrip memeriksa secara otomatis bahwa aturan di atas terpenuhi.

| Fold | Test | Validation | Menit test | Menit validation | Menit train |
|---|---|---|---|---|---|
| 1 | S03, S07, S14, S17 | S05, S13 | 38,9 | 21,4 | 135,6 |
| 2 | S05, S10, S11, S13 | S02, S18 | 43,6 | 16,3 | 136,1 |
| 3 | S02, S06, S15, S18 | S01, S19 | 37,8 | 19,0 | 139,2 |
| 4 | S01, S09, S16, S19 | S04, S20 | 39,6 | 17,2 | 139,2 |
| 5 | S04, S08, S12, S20 | S03, S17 | 36,1 | 19,2 | 140,7 |

Berkas: `data/splits/speaker_folds.csv` (peran tiap penutur pada tiap fold) dan `data/splits/fold1.csv` sampai `fold5.csv` (`utt_id`, `speaker`, `split`, mencakup seluruh baris manifest termasuk yang bertanda).

Menit pada tabel dihitung dari segmen berteks dan berdurasi positif, sebelum aturan filter apa pun.

## Evaluasi pendamping

Hasil utama memakai semua segmen usable (3.090). Sebagai pembanding, `src/evaluate.py` juga menghitung **subset bersih** yang mengeluarkan segmen berikut (satu segmen bisa punya beberapa alasan; aturan di `src/data.py`):

| Alasan | Segmen |
|---|---|
| Durasi < 1 s | 257 |
| Berlabel Indonesia atau Inggris (k11) | 185 |
| Laju teks > 25 karakter/s (kemungkinan tidak selaras) | 33 |
| Referensi ber-tag `[...]` | 31 |
| Bertanda hening digital atau halusinasi ASR | 21 |
| Penutur membacakan pertanyaan panduan | 12 |
| Referensi memuat angka | 8 |
| **Tersisa di subset bersih** | **2.592** |

Label dialek kedua korpus berbeda konvensi, sehingga analisis dialek memakai label tingkat penutur (`data/speakers.csv`) yang dikelompokkan menjadi Timuran, Semarangan, Solo-Jogja (Mataraman), Campuran, dan Tidak dirinci.

## Keterbatasan

- Hanya 4 penutur test per fold, sehingga skor per fold sangat dipengaruhi penutur tertentu. Laporkan rata-rata dan simpangan baku antar fold, dan sebaiknya juga skor per penutur.
- Dialek tidak seimbang: k05 didominasi Jawa Jogja (6 dari 10 penutur), sedangkan Jawa Timuran hanya S01, S07, S08, S10, dan S14.
- Pada anotasi k11, hanya S03, S05, dan S09 yang berlabel non-asli atau heritage.
- Transkripsi k11 masih draf ASR dan transkripsi k05 belum diketahui status review-nya. Hasil evaluasi tidak bermakna sebelum transkripsi dikoreksi.
- Konvensi transkripsi kedua korpus berbeda (misalnya ejaan fonetis dan tag `[NAMA]` pada k05), belum diseragamkan.
- Panjang segmen berbeda jauh (rata-rata k11 sekitar 2,7 detik, k05 sekitar 6,5 detik) dan belum digabung menjadi potongan untuk model.
- Penutur menyebut namanya sendiri pada bagian perkenalan dan nama itu muncul di `text_raw` sebagian segmen. Teks dibiarkan apa adanya karena penutur tidak keberatan.
- Topik berbeda antar korpus (k11: pertanyaan Makanan dan Pendidikan, k05: topik Perjalanan) dan label topik belum diselaraskan.

## Cara membuat ulang

```bash
./scripts/fetch_data.sh
python3 src/build_manifest.py
python3 src/make_splits.py
python3 src/normalize_text.py --tokenize
```

Jika korpus sudah ada di folder lain pada commit yang sama, set `CORPUS_K11` dan `CORPUS_K05` sebagai pengganti `fetch_data.sh`.
