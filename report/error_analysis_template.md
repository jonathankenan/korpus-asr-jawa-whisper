# Templat Laporan Analisis Kesalahan (SPEC.md §5C)

Gunakan templat ini untuk merangkum temuan dari `results/error_analysis_<tag>.md`:

## 1. Kata yang Sering Salah Dikenali
- **Substitusi vokal/dialek** (misal: `o` ↔ `a` pada *sego* / *sega*, *piro* / *pira*).
- **Kata fungsi & partikel Jawa** (misal: *sing*, *ning*, *ing*, *ta*, *kok*, *lha*).

## 2. Pengaruh Variasi Dialek & Aksen
- Bandingkan WER/CER antara **Jawa Timuran (Suroboyoan)**, **Jawa Jogja/Solo**, dan **Semarangan**.
- Jelaskan ketidakseimbangan distribusi dialek (dominasi Jawa Jogja dari `corpus_k05`).

## 3. Pengaruh Status Penutur
- Bandingkan performa **native** vs **non-native** (`S03, S09, S11, S12`) vs **heritage** (`S05`).

## 4. Pengaruh Noise & Kualitas Rekaman
- Tinjau segmen yang mengandung `event` atau durasi pendek (`short_lt1s`).

## 5. Kata Jarang (OOV) & Campur Kode
- Evaluasi kesalahan pada ujaran **Campur (Jawa-Indonesia)** dan **Campur (Jawa-Inggris)**.
- Evaluasi perbedaan bahasa lisan (ejaan fonetis, jeda *eee*, pengulangan terpotong `[TRUNCATED]`) dengan bentuk penulisan baku.
