"""Menyatukan korpus kelompok ini (corpus_k11) dan kelompok 5 (corpus_k05) menjadi satu manifest.

Keluaran (folder data/):
  manifest.csv    satu baris per segmen anotasi, semua baris dipertahankan
  speakers.csv    satu baris per penutur gabungan (S01-S10 kelompok ini, S11-S20 kelompok 5)
  id_mapping.csv  pemetaan ID penutur baru ke ID asal

Skrip ini TIDAK menyaring segmen dan TIDAK menormalisasi teks. Segmen bermasalah hanya diberi
tanda pada kolom `flags`, dan kolom `usable` menandai baris yang tidak mungkin dipakai.
Label bahasa dikoreksi memakai LABEL_FIXES di bawah (label asal tetap disimpan di
`language_label_src`).
Kolom label_speaker pada anotasi kelompok ini tidak disalin karena memuat nama penutur.

Jalankan dari akar repo setelah scripts/fetch_data.sh:
  python3 src/build_manifest.py
"""
import csv
import collections
import wave
from pathlib import Path

from audio import corpus_dir, resolve_audio_path

ROOT = Path(__file__).resolve().parent.parent
K11 = corpus_dir("corpus_k11")  # default corpus_k11/, dapat dialihkan lewat env CORPUS_K11
K05 = corpus_dir("corpus_k05")  # default corpus_k05/, dapat dialihkan lewat env CORPUS_K05
OUT = ROOT / "data"

K05_ID_OFFSET = 10  # S01-S10 kelompok 5 menjadi S11-S20

SHORT_SEC = 1.0
LONG_SEC = 30.0
BEYOND_TOL = 0.05
UNUSABLE_FLAGS = {"no_text", "neg_duration", "beyond_audio", "audio_missing"}

# Koreksi label bahasa k11: segmen berlabel Inggris padahal teksnya tidak memuat kata Inggris.
# utt_id -> (label asal, label baru)
LABEL_FIXES = {
    "S02_0066": ("Campur (Jawa-Inggris)", "Jawa"),
    "S02_0153": ("Campur (Jawa-Inggris)", "Indonesia"),
    "S05_0081": ("Campur (Jawa-Inggris)", "Campur (Jawa-Indonesia)"),
    "S07_0289": ("Inggris", "Jawa"),
    "S07_0411": ("Inggris", "Indonesia"),
    "S07_0508": ("Inggris", "Jawa"),
    "S07_0509": ("Inggris", "Jawa"),
    "S07_0510": ("Inggris", "Jawa"),
    "S07_0511": ("Inggris", "Jawa"),
    "S10_0164": ("Campur (Jawa-Inggris)", "Jawa"),
    "S10_0172": ("Campur (Jawa-Inggris)", "Campur (Jawa-Indonesia)"),
    "S10_0247": ("Campur (Jawa-Inggris)", "Jawa"),
    "S10_0275": ("Campur (Jawa-Inggris)", "Jawa"),
    "S10_0324": ("Campur (Jawa-Inggris)", "Campur (Jawa-Indonesia)"),
}

_dur_cache = {}


def wav_seconds(path):
    key = str(path)
    if key not in _dur_cache:
        if not path.exists():
            _dur_cache[key] = None
        else:
            with wave.open(key) as w:
                _dur_cache[key] = w.getnframes() / w.getframerate()
    return _dur_cache[key]


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def to_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def gender_code(g):
    g = (g or "").strip().lower()
    return {"m": "M", "male": "M", "f": "F", "female": "F"}.get(g, "")


def clean_dialect(label):
    label = (label or "").strip()
    if label in ("", "-"):
        return ""
    return label.split(" – ")[0].strip()


def status_from_dialect_label(label):
    label = label or ""
    if "non-asli" in label:
        return "non-native"
    if "warisan" in label:
        return "heritage"
    return "native"


def row_flags(start, end, text, audio_sec, audio_exists, fixed_name):
    flags = []
    if not text:
        flags.append("no_text")
    if start is None or end is None or end <= start:
        flags.append("neg_duration")
    else:
        d = end - start
        if d < SHORT_SEC:
            flags.append("short_lt1s")
        if d > LONG_SEC:
            flags.append("long_gt30s")
    if not audio_exists:
        flags.append("audio_missing")
    elif end is not None and audio_sec is not None and end > audio_sec + BEYOND_TOL:
        flags.append("beyond_audio")
    if fixed_name:
        flags.append("filename_fixed")
    return flags


def build():
    manifest = []
    counters = collections.Counter()

    # --- kelompok ini (k11) ---
    meta11 = {r["Speaker"]: r for r in read_csv(K11 / "metadata" / "metadata.csv")}
    ann11 = read_csv(K11 / "annotation" / "annotation-semua.csv")
    dial11 = collections.defaultdict(collections.Counter)
    for r in ann11:
        d = (r["label_dialek"] or "").strip()
        if d and d != "-":
            dial11[r["file"][:3]][d] += 1
    spk11 = {}
    for spk, c in dial11.items():
        label = c.most_common(1)[0][0]
        spk11[spk] = {
            "dialect": clean_dialect(label),
            "status": status_from_dialect_label(label),
        }

    for r in ann11:
        orig_file = r["file"]
        spk = orig_file[:3]
        audio = K11 / "audio" / spk / orig_file
        seg = r["segment_id"].strip()
        text = (r["transkripsi"] or "").strip()
        if text == "-":
            text = ""
        start, end = to_float(r["start"]), to_float(r["end"])
        counters[spk] += 1
        flags = row_flags(start, end, text, wav_seconds(audio), audio.exists(), False)
        if seg.endswith("gap") and "no_text" not in flags:
            flags.append("no_text")
        manifest.append({
            "utt_id": f"{spk}_{counters[spk]:04d}",
            "speaker": spk,
            "group": "k11",
            "orig_speaker": spk,
            "orig_file": orig_file,
            "orig_seg": seg,
            "audio_path": f"corpus_k11/{audio.relative_to(K11).as_posix()}",
            "start": r["start"],
            "end": r["end"],
            "duration": round(end - start, 3) if start is not None and end is not None else "",
            "text_raw": text,
            "language_label": "" if (r["label_bahasa"] or "").strip() == "-" else (r["label_bahasa"] or "").strip(),
            "dialect": clean_dialect(r["label_dialek"]),
            "topic": (r["label_topik"] or "").strip(),
            "event": (r["label_kejadian"] or "").strip(),
            "asr_confidence": (r["confidence_asr"] or "").strip(),
            "gender": gender_code(meta11[spk]["Gender"]),
            "speaker_status": spk11[spk]["status"],
            "transcript_status": "draft_asr",
            "flags": ";".join(flags),
        })

    # --- kelompok 5 (k05) ---
    meta05 = {}
    for r in read_csv(K05 / "metadata" / "metadata.csv"):
        meta05.setdefault(r["Speaker"], r)
    ann05 = read_csv(K05 / "annotation" / "annotation.csv")
    for r in ann05:
        orig_spk = r["Speaker_ID"].strip()
        spk = f"S{int(orig_spk[1:]) + K05_ID_OFFSET:02d}"
        orig_file = r["File"].strip()
        audio = K05 / "audio" / orig_spk / orig_file
        fixed = False
        if not audio.exists():
            alt = K05 / "audio" / orig_spk / orig_file.replace("_", "-")
            if alt.exists():
                audio, fixed = alt, True
        text = (r["Transcription"] or "").strip()
        start, end = to_float(r["Start_Time"]), to_float(r["End_Time"])
        counters[spk] += 1
        flags = row_flags(start, end, text, wav_seconds(audio), audio.exists(), fixed)
        m = meta05[orig_spk]
        manifest.append({
            "utt_id": f"{spk}_{counters[spk]:04d}",
            "speaker": spk,
            "group": "k05",
            "orig_speaker": orig_spk,
            "orig_file": orig_file,
            "orig_seg": "",
            "audio_path": f"corpus_k05/{audio.relative_to(K05).as_posix()}",
            "start": r["Start_Time"],
            "end": r["End_Time"],
            "duration": round(end - start, 3) if start is not None and end is not None else "",
            "text_raw": text,
            "language_label": "",
            "dialect": (r["Language_Dialect"] or "").strip(),
            "topic": (r["Topic"] or "").strip(),
            "event": (r["Event_Label"] or "").strip(),
            "asr_confidence": "",
            "gender": gender_code(m["Gender"]),
            "speaker_status": "non-native" if m["Speaker_Type"].strip().lower().startswith("non") else "native",
            "transcript_status": "unverified",
            "flags": ";".join(flags),
        })

    return manifest, spk11, meta05


def finalize(manifest):
    """Terapkan koreksi label bahasa (LABEL_FIXES) dan hitung kolom usable.

    language_label_src menyimpan label asal dari anotasi. usable = "no" untuk baris yang tidak
    mungkin dipakai (tanpa teks, waktu tidak valid, melewati akhir audio, atau audio hilang);
    tanda lain (short_lt1s, long_gt30s, filename_fixed) tidak memengaruhi usable.
    """
    seen = set()
    out = []
    for r in manifest:
        new = {}
        for k, v in r.items():
            new[k] = v
            if k == "language_label":
                new["language_label_src"] = v
        fix = LABEL_FIXES.get(r["utt_id"])
        if fix:
            old, repl = fix
            if r["language_label"] != old:
                raise SystemExit(f"koreksi label {r['utt_id']}: label asal {r['language_label']!r} "
                                 f"tidak sama dengan {old!r}")
            new["language_label"] = repl
            seen.add(r["utt_id"])
        flags = set(r["flags"].split(";")) - {""}
        new["usable"] = "no" if flags & UNUSABLE_FLAGS else "yes"
        out.append(new)
    missing = set(LABEL_FIXES) - seen
    if missing:
        raise SystemExit(f"utt_id pada LABEL_FIXES tidak ditemukan di manifest: {sorted(missing)}")
    print(f"koreksi label bahasa diterapkan: {len(seen)} baris")
    return out


def write_csv(path, rows, fields):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    OUT.mkdir(exist_ok=True)
    manifest, spk11, meta05 = build()
    manifest = finalize(manifest)
    fields = list(manifest[0].keys())
    write_csv(OUT / "manifest.csv", manifest, fields)

    # speakers.csv dan id_mapping.csv
    by_spk = collections.defaultdict(list)
    for r in manifest:
        by_spk[r["speaker"]].append(r)
    speakers, mapping = [], []
    for spk in sorted(by_spk):
        rows = by_spk[spk]
        g = rows[0]["group"]
        audio_files = sorted({r["audio_path"] for r in rows})
        audio_sec = sum(wav_seconds(resolve_audio_path(p)) or 0 for p in audio_files)
        usable = [r for r in rows if r["text_raw"] and "neg_duration" not in r["flags"]]
        dialect = spk11[spk]["dialect"] if g == "k11" else meta05[rows[0]["orig_speaker"]]["Language_Dialect"].strip()
        speakers.append({
            "speaker": spk,
            "group": g,
            "orig_speaker": rows[0]["orig_speaker"],
            "gender": rows[0]["gender"],
            "dialect": dialect,
            "speaker_status": rows[0]["speaker_status"],
            "n_segments": len(rows),
            "n_with_text": sum(1 for r in rows if r["text_raw"]),
            "audio_files": len(audio_files),
            "audio_seconds": round(audio_sec, 2),
            "transcribed_seconds": round(sum(float(r["duration"]) for r in usable), 2),
        })
        mapping.append({"speaker": spk, "group": g, "orig_speaker": rows[0]["orig_speaker"]})
    write_csv(OUT / "speakers.csv", speakers, list(speakers[0].keys()))
    write_csv(OUT / "id_mapping.csv", mapping, ["speaker", "group", "orig_speaker"])

    # ringkasan
    print(f"manifest.csv: {len(manifest)} baris, {len(by_spk)} penutur")
    for g in ("k11", "k05"):
        sp = [s for s in speakers if s["group"] == g]
        print(f"  {g}: {sum(s['n_segments'] for s in sp)} segmen, "
              f"audio {sum(s['audio_seconds'] for s in sp)/60:.1f} menit, "
              f"bertranskripsi {sum(s['transcribed_seconds'] for s in sp)/60:.1f} menit")
    fc = collections.Counter(f for r in manifest for f in r["flags"].split(";") if f)
    print("jumlah baris per flag:", dict(fc))
    print("usable:", dict(collections.Counter(r["usable"] for r in manifest)))
    for g in ("k11", "k05"):
        fc = collections.Counter(f for r in manifest if r["group"] == g for f in r["flags"].split(";") if f)
        print(f"  flag {g}:", dict(fc))


if __name__ == "__main__":
    main()
