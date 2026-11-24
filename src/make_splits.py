"""Membuat 5-fold split antar penutur dari data/manifest.csv dan data/speakers.csv.

Aturan:
  - tiap penutur menjadi test tepat satu kali;
  - tiap fold: 4 penutur test (2 laki-laki + 2 perempuan, 2 dari k11 dan 2 dari k05),
    2 penutur validation (1 laki-laki + 1 perempuan, 1 dari k11 dan 1 dari k05),
    14 penutur train;
  - tidak ada penutur yang muncul pada lebih dari satu subset dalam satu fold.

Penutur validation fold k diambil dari penutur test fold berikutnya (rotasi), sehingga tidak
memerlukan aturan pemilihan acak. Susunan fold di bawah adalah usulan awal; penutur non-native
atau warisan (lihat speakers.csv) disebar agar tidak menumpuk pada satu fold.

Keluaran (folder data/splits/):
  speaker_folds.csv   peran tiap penutur (train/val/test) pada tiap fold
  fold1.csv ... fold5.csv   utt_id, speaker, split untuk seluruh baris manifest

Baris bertanda (kolom flags) tidak disaring di sini.

Jalankan dari akar repo: python3 src/make_splits.py
"""
import csv
import collections
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = DATA / "splits"

TEST = {
    1: ["S03", "S07", "S14", "S17"],
    2: ["S05", "S10", "S11", "S13"],
    3: ["S02", "S06", "S15", "S18"],
    4: ["S01", "S09", "S16", "S19"],
    5: ["S04", "S08", "S12", "S20"],
}


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main():
    speakers = {r["speaker"]: r for r in read_csv(DATA / "speakers.csv")}
    manifest = read_csv(DATA / "manifest.csv")
    folds = sorted(TEST)

    # validation: 1 laki-laki k11 + 1 perempuan k05 dari test fold berikutnya
    val = {}
    for k in folds:
        nxt = TEST[k % len(folds) + 1]
        m_k11 = [s for s in nxt if speakers[s]["group"] == "k11" and speakers[s]["gender"] == "M"]
        f_k05 = [s for s in nxt if speakers[s]["group"] == "k05" and speakers[s]["gender"] == "F"]
        val[k] = m_k11 + f_k05

    # --- validasi aturan ---
    all_test = [s for k in folds for s in TEST[k]]
    assert sorted(all_test) == sorted(speakers), "setiap penutur harus menjadi test tepat sekali"
    for k in folds:
        t, v = TEST[k], val[k]
        assert len(t) == 4 and len(v) == 2, (k, t, v)
        assert not set(t) & set(v), f"fold {k}: test dan val beririsan"
        g = collections.Counter((speakers[s]["group"], speakers[s]["gender"]) for s in t)
        assert g == {("k11", "M"): 1, ("k11", "F"): 1, ("k05", "M"): 1, ("k05", "F"): 1}, (k, g)
        gv = collections.Counter(speakers[s]["gender"] for s in v)
        assert gv == {"M": 1, "F": 1}, (k, gv)

    role = {k: {} for k in folds}
    for k in folds:
        for s in speakers:
            role[k][s] = "test" if s in TEST[k] else "val" if s in val[k] else "train"
        assert collections.Counter(role[k].values()) == {"train": 14, "val": 2, "test": 4}, k

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "speaker_folds.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["speaker", "group", "gender", "dialect", "speaker_status"] + [f"fold{k}" for k in folds])
        for s in sorted(speakers):
            sp = speakers[s]
            w.writerow([s, sp["group"], sp["gender"], sp["dialect"], sp["speaker_status"]] + [role[k][s] for k in folds])

    for k in folds:
        with open(OUT / f"fold{k}.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["utt_id", "speaker", "split"])
            for r in manifest:
                w.writerow([r["utt_id"], r["speaker"], role[k][r["speaker"]]])

    # ringkasan: menit bertranskripsi (baris bernaskah, durasi positif) per subset
    usable = collections.defaultdict(float)
    for r in manifest:
        if r["text_raw"] and "neg_duration" not in r["flags"]:
            usable[r["speaker"]] += float(r["duration"])
    print("fold | test (menit) | val (menit) | train (menit) | penutur test")
    for k in folds:
        mins = {p: sum(usable[s] for s in speakers if role[k][s] == p) / 60 for p in ("test", "val", "train")}
        print(f"  {k}  | {mins['test']:6.1f}       | {mins['val']:5.1f}       | {mins['train']:6.1f}        | "
              f"{', '.join(TEST[k])} | val: {', '.join(val[k])}")
    print("OK: semua pemeriksaan lolos")


if __name__ == "__main__":
    main()
