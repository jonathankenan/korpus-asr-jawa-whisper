"""Analisis kesalahan transkripsi ASR Bahasa Jawa (spesifikasi Fase 2 §5C).

Menganalisis:
  1. Komposisi kesalahan (substitusi, penghapusan, penyisipan) dan kata tersering.
  2. Perbedaan bahasa lisan vs tulis (varian ejaan a/o, dh/d, th/t, e/i, u/o).
  3. Kata jarang: tingkat salah menurut frekuensi kata pada data latih fold tersebut.
  4. Pengaruh penutur, dialek, status penutur, kejadian/noise, durasi, dan bahasa.
  5. Kegagalan decoding (prediksi kosong, penyisipan berlebih) dan ujaran terburuk.
  6. Opsional --baseline: kata yang membaik atau memburuk dibanding eksperimen lain.

Contoh:
  python src/analyze_errors.py --exp small-pretrained
  python src/analyze_errors.py --exp tiny-ft --baseline tiny-pretrained
  python src/analyze_errors.py --self-test
"""

import argparse
import collections
from pathlib import Path
from typing import Dict, List, Tuple

from data import load_manifest, split_map
from evaluate import aggregate, load_predictions, md_cell, table
from normalize_text import normalize

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "results" / "analysis"
FREQ_BUCKETS = [(0, 0, "tidak ada di data latih (OOV)"), (1, 5, "1-5 kali"),
                (6, 50, "6-50 kali"), (51, 10 ** 9, "> 50 kali")]
# (nama, pengganti) -> dua kata dianggap varian jika sama setelah penggantian
SPELLING_RULES = [("a/o", (("o", "a"),)), ("dh/d", (("dh", "d"),)), ("th/t", (("th", "t"),)),
                  ("e/i", (("i", "e"),)), ("o/u", (("u", "o"),))]


def align_words(ref: List[str], hyp: List[str]) -> List[Tuple[str, str, str]]:
    """Sejajarkan kata ref dan hyp. Kembalikan daftar (op, ref_tok, hyp_tok)."""
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    back = [[""] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = i
        back[i][0] = "del"
    for j in range(1, m + 1):
        dp[0][j] = j
        back[0][j] = "ins"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                back[i][j] = "ok"
            else:
                choices = [
                    (dp[i - 1][j - 1] + 1, "sub"),
                    (dp[i - 1][j] + 1, "del"),
                    (dp[i][j - 1] + 1, "ins"),
                ]
                dp[i][j], back[i][j] = min(choices, key=lambda x: x[0])

    ops: List[Tuple[str, str, str]] = []
    i, j = n, m
    while i > 0 or j > 0:
        b = back[i][j]
        if b in ("ok", "sub"):
            ops.append((b, ref[i - 1], hyp[j - 1]))
            i -= 1
            j -= 1
        elif b == "del":
            ops.append(("del", ref[i - 1], "<eps>"))
            i -= 1
        else:
            ops.append(("ins", "<eps>", hyp[j - 1]))
            j -= 1
    ops.reverse()
    return ops


def spelling_variant(ref: str, hyp: str) -> str:
    """Nama aturan jika dua kata hanya berbeda ejaan, selain itu string kosong."""
    for name, pairs in SPELLING_RULES:
        a, b = ref, hyp
        for old, new in pairs:
            a, b = a.replace(old, new), b.replace(old, new)
        if a == b:
            return name
    squeeze = lambda w: "".join(c for i, c in enumerate(w) if i == 0 or c != w[i - 1])
    return "huruf ganda" if squeeze(ref) == squeeze(hyp) else ""


def train_vocabulary() -> Dict[int, collections.Counter]:
    """Frekuensi kata pada data latih tiap fold."""
    manifest = load_manifest()
    vocab = {}
    for k in range(1, 6):
        smap = split_map(k)
        vocab[k] = collections.Counter(
            w for r in manifest if smap[r["utt_id"]] == "train"
            for w in normalize(r["text_raw"]).split())
    return vocab


def collect(rows: List[Dict], vocab: Dict[int, collections.Counter]) -> Dict:
    """Hitung semua statistik kata dari baris prediksi."""
    subs, dels, inss = collections.Counter(), collections.Counter(), collections.Counter()
    variants, variant_examples = collections.Counter(), collections.defaultdict(collections.Counter)
    freq = collections.defaultdict(lambda: [0, 0])      # bucket -> [salah, total]
    per_word = collections.defaultdict(lambda: [0, 0])  # kata referensi -> [salah, total]
    n_sub = 0
    for r in rows:
        for op, rt, ht in align_words(r["ref"].split(), r["hyp"].split()):
            if op == "sub":
                subs[(rt, ht)] += 1
                n_sub += 1
                rule = spelling_variant(rt, ht)
                if rule:
                    variants[rule] += 1
                    variant_examples[rule][(rt, ht)] += 1
            elif op == "del":
                dels[rt] += 1
            elif op == "ins":
                inss[ht] += 1
            if op != "ins":
                wrong = int(op != "ok")
                per_word[rt][0] += wrong
                per_word[rt][1] += 1
                count = vocab[r["fold"]][rt] if vocab else 0
                label = next(name for lo, hi, name in FREQ_BUCKETS if lo <= count <= hi)
                freq[label][0] += wrong
                freq[label][1] += 1
    return {"subs": subs, "dels": dels, "inss": inss, "variants": variants,
            "variant_examples": variant_examples, "freq": freq, "per_word": per_word, "n_sub": n_sub}


def grouped_metrics(rows: List[Dict], key) -> List[List]:
    groups = collections.defaultdict(list)
    for r in rows:
        for value in key(r):
            groups[value].append(r)
    body = []
    for value, subset in sorted(groups.items()):
        m = aggregate(subset)
        body.append([value, m["n_utterances"], m["ref_words"], m["wer_pct"], m["cer_pct"]])
    return body


def write_report(exp: str, rows: List[Dict], baseline: str, top_k: int, out_path: Path) -> None:
    vocab = train_vocabulary()
    by_group = {g: [r for r in rows if r["group"] == g] for g in ("k05", "k11")}
    stats = {g: collect(sel, vocab) for g, sel in by_group.items()}
    lines = [f"# Analisis Kesalahan ASR ({exp})", "",
             "Dibuat otomatis oleh `src/analyze_errors.py` dari prediksi test. Referensi k11 masih draf ASR, "
             "jadi kedua korpus dipisahkan.", "", "## 1. Komposisi kesalahan", ""]
    body = []
    for g, sel in by_group.items():
        m = aggregate(sel)
        errors = m["word_sub"] + m["word_del"] + m["word_ins"]
        body.append([g, m["wer_pct"], m["cer_pct"]] +
                    [f"{m[k]} ({100 * m[k] / max(1, errors):.0f}%)" for k in ("word_sub", "word_del", "word_ins")])
    lines += table(["Korpus", "WER", "CER", "Substitusi", "Penghapusan", "Penyisipan"], body)

    lines += ["## 2. Kata yang paling sering salah dikenali", ""]
    for g in ("k05", "k11"):
        s = stats[g]
        lines += [f"### {g}: substitusi tersering", ""]
        lines += table(["Referensi", "Prediksi", "Frekuensi"],
                       [[f"`{a}`", f"`{b}`", c] for (a, b), c in s["subs"].most_common(top_k)])
        lines += [f"### {g}: penghapusan dan penyisipan tersering", ""]
        dels, inss = s["dels"].most_common(top_k), s["inss"].most_common(top_k)
        lines += table(["Dihapus", "n", "Disisipkan", "n"],
                       [[f"`{dels[i][0]}`" if i < len(dels) else "", dels[i][1] if i < len(dels) else "",
                         f"`{inss[i][0]}`" if i < len(inss) else "", inss[i][1] if i < len(inss) else ""]
                        for i in range(max(len(dels), len(inss)))])
        hard = sorted(((w, e, t) for w, (e, t) in s["per_word"].items() if t >= 10),
                      key=lambda x: (-x[1] / x[2], -x[2], x[0]))[:top_k]
        lines += [f"### {g}: kata (muncul ≥ 10 kali) dengan tingkat salah tertinggi", ""]
        lines += table(["Kata", "Salah", "Muncul", "Tingkat salah (%)"],
                       [[f"`{w}`", e, t, round(100 * e / t)] for w, e, t in hard])

    lines += ["## 3. Perbedaan bahasa lisan dan bentuk tulis", "",
              "Substitusi yang hanya berbeda ejaan (lafal sama atau hampir sama).", ""]
    body = []
    for g in ("k05", "k11"):
        s = stats[g]
        total = sum(s["variants"].values())
        for rule, count in s["variants"].most_common():
            ex = ", ".join(f"{a}→{b} ({c})" for (a, b), c in s["variant_examples"][rule].most_common(4))
            body.append([g, rule, count, f"{100 * count / max(1, s['n_sub']):.1f}", ex])
        body.append([g, "**total varian**", total, f"{100 * total / max(1, s['n_sub']):.1f}", ""])
    lines += table(["Korpus", "Aturan", "Jumlah", "% dari substitusi", "Contoh (referensi→prediksi)"], body)

    lines += ["## 4. Kata jarang", "",
              "Tingkat salah kata referensi (substitusi + penghapusan) menurut frekuensinya pada data latih fold.", ""]
    body = []
    for _, _, name in FREQ_BUCKETS:
        row = [name]
        for g in ("k05", "k11"):
            e, t = stats[g]["freq"].get(name, [0, 0])
            row += [t, round(100 * e / t, 1) if t else "-"]
        body.append(row)
    lines += table(["Frekuensi di data latih", "Kata k05", "Salah k05 (%)", "Kata k11", "Salah k11 (%)"], body)

    header = ["Nilai", "Ujaran", "Kata", "WER (%)", "CER (%)"]
    sections = [
        ("5. Pengaruh penutur", lambda r: [f"{r['speaker']} ({r['group']}, {r['gender']}, {r['speaker_status']})"]),
        ("6. Pengaruh dialek (tingkat penutur)", lambda r: [f"{r['group']}: {r['dialect_family']}"]),
        ("7. Pengaruh status penutur", lambda r: [f"{r['group']}: {r['speaker_status']}"]),
        ("8. Pengaruh noise dan kejadian rekaman", lambda r: [f"{r['group']}: {e}" for e in r["events"]]),
        ("9. Pengaruh durasi ujaran", lambda r: [f"{r['group']}: {r['duration_bucket']}"]),
        ("10. Pengaruh campur kode (label bahasa k11)",
         lambda r: [r["language_label"]] if r["group"] == "k11" else []),
    ]
    for title, key in sections:
        lines += [f"## {title}", ""] + table(header, grouped_metrics(rows, key))

    lines += ["## 11. Kegagalan decoding", ""]
    body = []
    for g, sel in by_group.items():
        empty = sum(1 for r in sel if not r["hyp"])
        runaway = [r for r in sel if r["wi"] > r["wn"]]
        short = sum(1 for r in runaway if r["duration"] < 1)
        all_ins = sum(r["wi"] for r in sel)
        body.append([g, empty, len(runaway), short,
                     f"{100 * sum(r['wi'] for r in runaway) / max(1, all_ins):.0f}"])
    lines += table(["Korpus", "Prediksi kosong", "Penyisipan > jumlah kata referensi",
                    "… di antaranya < 1 s", "% seluruh penyisipan dari ujaran ini"], body)

    lines += ["## 12. Ujaran dengan WER tertinggi (referensi ≥ 4 kata)", ""]
    worst = sorted((r for r in rows if r["wn"] >= 4),
                   key=lambda r: (-(r["ws"] + r["wd"] + r["wi"]) / r["wn"], r["utt_id"]))[:top_k]
    lines += table(["utt_id", "WER (%)", "Durasi (s)", "Referensi", "Prediksi"],
                   [[r["utt_id"], round(100 * (r["ws"] + r["wd"] + r["wi"]) / r["wn"]), r["duration"],
                     md_cell(r["ref"][:200]), md_cell(r["hyp"][:200])] for r in worst])

    if baseline:
        base_rows = {r["utt_id"]: r for r in load_predictions(baseline)}
        shared = [r for r in rows if r["utt_id"] in base_rows]
        lines += [f"## 13. Perubahan dibanding `{baseline}`", ""]
        body = []
        for g in ("k05", "k11"):
            new = [r for r in shared if r["group"] == g]
            old = [base_rows[r["utt_id"]] for r in new]
            if not new:
                continue
            err = lambda r: r["ws"] + r["wd"] + r["wi"]
            better = sum(1 for a, b in zip(new, old) if err(a) < err(b))
            worse = sum(1 for a, b in zip(new, old) if err(a) > err(b))
            body.append([g, len(new), aggregate(old)["wer_pct"], aggregate(new)["wer_pct"], better, worse,
                         len(new) - better - worse])
        lines += table(["Korpus", "Ujaran", "WER pembanding", f"WER {exp}", "Membaik", "Memburuk", "Tetap"], body)
        for g in ("k05", "k11"):
            ids = {r["utt_id"] for r in shared if r["group"] == g}
            old_words = collect([base_rows[i] for i in sorted(ids)], vocab)["per_word"]
            new_words = collect([r for r in shared if r["group"] == g], vocab)["per_word"]
            delta = sorted(((w, old_words[w][0], new_words[w][0], t) for w, (_, t) in new_words.items()),
                           key=lambda x: (x[2] - x[1], x[0]))
            better = [x for x in delta if x[2] < x[1]][:top_k]
            worse = [x for x in reversed(delta) if x[2] > x[1]][:top_k]
            for title, items in (("paling membaik", better), ("paling memburuk", worse)):
                lines += [f"### {g}: kata {title}", ""]
                lines += table(["Kata", "Salah (pembanding)", f"Salah ({exp})", "Muncul"],
                               [[f"`{w}`", a, b, t] for w, a, b, t in items])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_self_test():
    ops = align_words("aku mangan sego goreng".split(), "aku mangan sega goreng".split())
    assert [o for o in ops if o[0] != "ok"] == [("sub", "sego", "sega")]
    ops = align_words("sugeng rawuh ing bandung".split(), "sugeng rawuh bandung".split())
    assert [o for o in ops if o[0] != "ok"] == [("del", "ing", "<eps>")]
    assert spelling_variant("sego", "sega") == "a/o"
    assert spelling_variant("wedhi", "wedi") == "dh/d"
    assert spelling_variant("mangan", "turu") == ""
    print("Self-test lolos: alignment dan deteksi varian ejaan akurat.")


def main():
    parser = argparse.ArgumentParser(description="Analisis kesalahan transkripsi ASR")
    parser.add_argument("--exp", default=None, help="Nama eksperimen di results/predictions")
    parser.add_argument("--baseline", default="", help="Eksperimen pembanding (opsional)")
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return
    if not args.exp:
        raise SystemExit("Berikan --exp <nama> atau --self-test.")

    rows = load_predictions(args.exp)
    if not rows:
        raise SystemExit(f"Tidak ada prediksi untuk {args.exp}.")
    out_path = OUT_DIR / f"{args.exp}.md"
    write_report(args.exp, rows, args.baseline, args.top_k, out_path)
    print(f"Laporan analisis kesalahan disimpan ke {out_path}")


if __name__ == "__main__":
    main()
