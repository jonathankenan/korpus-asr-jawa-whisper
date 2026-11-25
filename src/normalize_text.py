"""Normalisasi teks transkripsi korpus ASR Bahasa Jawa.

Masalah yang diatasi (sesuai DATASET.md dan spesifikasi Fase 2):

  1. Penyeragaman konvensi dua korpus:
       - k11: transkripsi draf ASR, bisa mengandung tanda baca bebas.
       - k05: menggunakan tag [NAMA], [MUSIC], [NOISE], dsb. dan ejaan fonetis.
  2. Normalisasi umum:
       a. Strip whitespace berlebih.
       b. Lowercase.
       c. Hapus tanda baca yang tidak relevan untuk ASR (koma, titik, tanda tanya,
          tanda seru, tanda hubung, petik, tanda kurung biasa).
       d. Pertahankan tanda hubung antar-morfem yang lazim dalam bahasa Jawa
          (opsional, dikendalikan flag --keep-hyphen).
  3. Penanganan tag khusus k05:
       - [NAMA], [NOISE], [MUSIC], [LAUGHTER], [BREATH], [COUGH], dll. → dihapus
         (bukan ujaran yang perlu dikenali).
       - Konten dalam kurung siku yang berisi kata non-Latin (misalnya [*]) → dihapus.
  4. Penyeragaman karakter:
       - Apostrof tipografis (', ') → apostrof standar (').
       - Tanda hubung panjang (–, —) → spasi.
       - Angka → dibiarkan (opsional: dapat di-expand, tetapi tidak dilakukan di sini
         karena konvensi numerik Bahasa Jawa belum baku).
  5. Kolom usable: hanya baris usable == "yes" yang diproses; sisanya disalin apa adanya.
  6. Tokenisasi karakter: opsional (--tokenize), menghasilkan kolom `text_chars`
     berisi daftar karakter dipisahkan spasi — format yang lazim dipakai Whisper
     dan CTC-based model.

Masukan : data/manifest.csv
Keluaran: data/manifest_normalized.csv (kolom tambahan: text_norm, text_chars)

Jalankan dari akar repo:
  python3 src/normalize_text.py [--manifest data/manifest.csv]
                                [--out data/manifest_normalized.csv]
                                [--keep-hyphen]
                                [--tokenize]
"""

import argparse
import csv
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Pola & konstanta
# ---------------------------------------------------------------------------

# Tag khusus k05 dan tag umum dalam tanda kurung siku → dihapus semuanya
_RE_BRACKET_TAG = re.compile(r"\[[^\]]*\]")

# Karakter tanda baca yang dihapus (tanpa apostrof, tanpa tanda hubung dulu)
_PUNCT_REMOVE = str.maketrans({
    ".": " ", ",": " ", "?": " ", "!": " ",
    "(": " ", ")": " ", '"': " ", """: " ", """: " ",
    "«": " ", "»": " ", "…": "...", "–": " ", "—": " ",
    # Karakter khas OCR/copy-paste
    "\u200b": "",  # zero-width space
    "\ufeff": "",  # BOM
})

# Apostrof tipografis → standar
_RE_APOS = re.compile(r"['\u2018\u2019\u02bc]")

# Spasi berlebih
_RE_SPACES = re.compile(r"\s+")

# Validasi: karakter yang diperbolehkan dalam output akhir
# (huruf Latin, angka, apostrof, tanda hubung, spasi)
_RE_ALLOWED = re.compile(r"[^a-z0-9'\- ]")


# ---------------------------------------------------------------------------
# Fungsi normalisasi
# ---------------------------------------------------------------------------

def remove_bracket_tags(text: str) -> str:
    """Hapus semua tag dalam tanda kurung siku, termasuk tag k05 seperti [NAMA]."""
    return _RE_BRACKET_TAG.sub(" ", text)


def unicode_normalize(text: str) -> str:
    """NFC normalization + ganti karakter apostrof tipografis."""
    text = unicodedata.normalize("NFC", text)
    text = _RE_APOS.sub("'", text)
    return text


def remove_punct(text: str, keep_hyphen: bool = False) -> str:
    """Hapus tanda baca. Jika keep_hyphen=True, tanda hubung dipertahankan."""
    text = text.translate(_PUNCT_REMOVE)
    if not keep_hyphen:
        text = text.replace("-", " ")
    return text


def remove_nonlatin(text: str) -> str:
    """Hapus karakter non-Latin yang tersisa (aksara asing, simbol, dll.)."""
    return _RE_ALLOWED.sub(" ", text)


def normalize(text: str, keep_hyphen: bool = False) -> str:
    """Pipeline normalisasi lengkap untuk satu string teks."""
    if not text or not text.strip():
        return ""
    text = unicode_normalize(text)
    text = remove_bracket_tags(text)
    text = text.lower()
    text = remove_punct(text, keep_hyphen=keep_hyphen)
    text = remove_nonlatin(text)
    text = _RE_SPACES.sub(" ", text).strip()
    return text


def char_tokenize(text: str) -> str:
    """Tokenisasi level karakter: setiap karakter dipisahkan spasi.

    Kata dipisahkan oleh '|' (konvensi Whisper/CTC).
    Contoh: "sugeng rawuh" → "s u g e n g | r a w u h"
    """
    if not text:
        return ""
    tokens = []
    for word in text.split():
        tokens.extend(list(word))
        tokens.append("|")
    if tokens and tokens[-1] == "|":
        tokens.pop()
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# Statistik
# ---------------------------------------------------------------------------

def collect_stats(rows_before, rows_after, key="text_norm"):
    """Hitung statistik perubahan normalisasi."""
    changed = sum(
        1 for a, b in zip(rows_before, rows_after)
        if a.get("text_raw", "") != b.get(key, "")
    )
    empty_after = sum(1 for b in rows_after if not b.get(key, ""))
    return {"changed": changed, "empty_after_norm": empty_after}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Normalisasi teks transkripsi korpus ASR Bahasa Jawa"
    )
    parser.add_argument("--manifest", default="data/manifest.csv")
    parser.add_argument("--out", default="data/manifest_normalized.csv")
    parser.add_argument(
        "--keep-hyphen", action="store_true",
        help="Pertahankan tanda hubung antar morfem dalam output"
    )
    parser.add_argument(
        "--tokenize", action="store_true",
        help="Tambahkan kolom text_chars (tokenisasi level karakter untuk CTC/Whisper)"
    )
    args = parser.parse_args()

    manifest_path = ROOT / args.manifest
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(manifest_path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print("Manifest kosong.")
        return

    fieldnames = list(rows[0].keys())
    # Sisipkan kolom baru setelah text_raw
    if "text_norm" not in fieldnames:
        idx = fieldnames.index("text_raw") + 1 if "text_raw" in fieldnames else len(fieldnames)
        fieldnames.insert(idx, "text_norm")
    if args.tokenize and "text_chars" not in fieldnames:
        idx = fieldnames.index("text_norm") + 1
        fieldnames.insert(idx, "text_chars")

    out_rows = []
    n_ok = 0
    n_skip = 0
    n_empty_result = 0

    for row in rows:
        new = dict(row)
        raw = row.get("text_raw", "")

        if row.get("usable", "yes") != "yes" or not raw.strip():
            # Baris tidak usable atau kosong: tetap normalisasi (bisa jadi hasilnya "")
            norm = normalize(raw, keep_hyphen=args.keep_hyphen)
            new["text_norm"] = norm
            if args.tokenize:
                new["text_chars"] = char_tokenize(norm)
            n_skip += 1
        else:
            norm = normalize(raw, keep_hyphen=args.keep_hyphen)
            new["text_norm"] = norm
            if args.tokenize:
                new["text_chars"] = char_tokenize(norm)
            if norm:
                n_ok += 1
            else:
                n_empty_result += 1

        out_rows.append(new)

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(out_rows)

    print(f"Normalisasi selesai → {args.out}")
    print(f"  Total baris      : {len(rows)}")
    print(f"  Usable, ada teks : {n_ok}")
    print(f"  Usable, hasil kosong setelah norm : {n_empty_result}")
    print(f"  Tidak usable / kosong             : {n_skip}")
    print(f"  Kolom baru: text_norm" + (", text_chars" if args.tokenize else ""))

    # Contoh hasil normalisasi (10 baris pertama dengan teks)
    print("\nContoh normalisasi (10 baris pertama dengan teks):")
    shown = 0
    for row in out_rows:
        if row.get("text_raw", "").strip() and shown < 10:
            print(f"  [{row['utt_id']}] raw : {row['text_raw'][:80]}")
            print(f"         norm: {row['text_norm'][:80]}")
            if args.tokenize:
                print(f"         chars: {row.get('text_chars','')[:80]}")
            shown += 1

    # Simpan ringkasan ke results/
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    summary_path = results_dir / "normalize_text_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Manifest input  : {args.manifest}\n")
        f.write(f"Output          : {args.out}\n")
        f.write(f"keep_hyphen     : {args.keep_hyphen}\n")
        f.write(f"tokenize        : {args.tokenize}\n")
        f.write(f"Total baris     : {len(rows)}\n")
        f.write(f"Usable+teks OK  : {n_ok}\n")
        f.write(f"Kosong setelah norm: {n_empty_result}\n")
        f.write(f"Tidak usable/kosong: {n_skip}\n")
    print(f"\nRingkasan disimpan ke {summary_path}")


if __name__ == "__main__":
    main()
