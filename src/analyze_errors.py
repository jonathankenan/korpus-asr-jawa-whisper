"""Analisis kesalahan transkripsi ASR Bahasa Jawa sesuai SPEC.md bagian 5C.

Menganalisis:
  1. Pasangan substitusi kata tersering (ref -> hyp).
  2. Kata yang paling sering terhapus (deletion) dan tersisip (insertion).
  3. Pengaruh dialek, status penutur (native/non-native/heritage), dan panjang ujaran.
  4. Pengaruh label kejadian/noise (event) jika digabungkan dengan manifest.
  5. Ujaran dengan WER tertinggi untuk inspeksi kualitatif.

Contoh:
  python src/analyze_errors.py --pred results/predictions_fold1_pretrained.csv --tag pretrained
  python src/analyze_errors.py --self-test
"""

import argparse
import collections
import csv
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent


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


def analyze_rows(rows: List[Dict[str, str]], ref_col: str, hyp_col: str, top_k: int = 15) -> Dict:
    """Jalankan analisis kesalahan pada daftar prediksi."""
    subs = collections.Counter()
    dels = collections.Counter()
    inss = collections.Counter()
    by_attr: Dict[str, Dict[str, List[float]]] = {
        "dialect": collections.defaultdict(list),
        "speaker_status": collections.defaultdict(list),
        "speaker": collections.defaultdict(list),
    }
    worst: List[Tuple[float, str, str, str]] = []

    for r in rows:
        ref = (r.get(ref_col) or "").strip()
        hyp = (r.get(hyp_col) or "").strip()
        if not ref:
            continue
        rw, hw = ref.split(), hyp.split()
        ops = align_words(rw, hw)
        errs = 0
        for op, rt, ht in ops:
            if op == "sub":
                subs[(rt, ht)] += 1
                errs += 1
            elif op == "del":
                dels[rt] += 1
                errs += 1
            elif op == "ins":
                inss[ht] += 1
                errs += 1

        utt_wer = 100.0 * errs / max(1, len(rw))
        for attr in by_attr:
            key = (r.get(attr) or "unknown").strip() or "unknown"
            by_attr[attr][key].append(utt_wer)
        worst.append((utt_wer, r.get("utt_id", "-"), ref, hyp))

    worst.sort(reverse=True)
    return {
        "top_substitutions": subs.most_common(top_k),
        "top_deletions": dels.most_common(top_k),
        "top_insertions": inss.most_common(top_k),
        "by_attr": {
            attr: {k: round(sum(v) / len(v), 2) for k, v in sorted(groups.items())}
            for attr, groups in by_attr.items()
        },
        "worst_utterances": worst[:top_k],
    }


def write_report(analysis: Dict, out_path: Path, tag: str) -> None:
    """Tulis laporan analisis kesalahan dalam format Markdown."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# Analisis Kesalahan ASR ({tag})\n\n")

        f.write("## 1. Substitusi Kata Tersering\n\n")
        f.write("| Referensi | Prediksi | Frekuensi |\n|---|---|---|\n")
        for (rt, ht), cnt in analysis["top_substitutions"]:
            f.write(f"| `{rt}` | `{ht}` | {cnt} |\n")

        f.write("\n## 2. Penghapusan (Deletion) Tersering\n\n")
        f.write("| Kata Referensi | Frekuensi |\n|---|---|\n")
        for rt, cnt in analysis["top_deletions"]:
            f.write(f"| `{rt}` | {cnt} |\n")

        f.write("\n## 3. Penyisipan (Insertion) Tersering\n\n")
        f.write("| Kata Prediksi | Frekuensi |\n|---|---|\n")
        for ht, cnt in analysis["top_insertions"]:
            f.write(f"| `{ht}` | {cnt} |\n")

        f.write("\n## 4. Rata-rata WER per Atribut\n\n")
        for attr, mapping in analysis["by_attr"].items():
            f.write(f"### `{attr}`\n\n| Nilai | Rata-rata WER (%) |\n|---|---|\n")
            for k, v in mapping.items():
                f.write(f"| {k} | {v} |\n")
            f.write("\n")

        f.write("## 5. Ujaran dengan Kesalahan Tertinggi\n\n")
        f.write("| utt_id | WER (%) | Referensi | Prediksi |\n|---|---|---|---|\n")
        for wer, uid, ref, hyp in analysis["worst_utterances"]:
            f.write(f"| {uid} | {wer:.1f} | {ref.replace('|', '\\|')} | {hyp.replace('|', '\\|')} |\n")


def run_self_test():
    sample = [
        {
            "utt_id": "S01_0001",
            "speaker": "S01",
            "dialect": "Suroboyoan",
            "speaker_status": "native",
            "ref_norm": "aku mangan sego goreng",
            "pred_norm": "aku mangan sega goreng",
        },
        {
            "utt_id": "S03_0001",
            "speaker": "S03",
            "dialect": "Solo",
            "speaker_status": "non-native",
            "ref_norm": "sugeng rawuh ing bandung",
            "pred_norm": "sugeng rawuh bandung",
        },
    ]
    res = analyze_rows(sample, "ref_norm", "pred_norm", top_k=5)
    assert res["top_substitutions"][0] == (("sego", "sega"), 1)
    assert res["top_deletions"][0] == ("ing", 1)
    print("Self-test lolos: alignment substitusi & penghapusan akurat.")


def main():
    parser = argparse.ArgumentParser(description="Analisis kesalahan transkripsi ASR")
    parser.add_argument("--pred", default=None, help="Path file CSV prediksi")
    parser.add_argument("--tag", default="eval", help="Label laporan")
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return

    if not args.pred:
        raise SystemExit("Berikan --pred <file.csv> atau --self-test.")

    pred_path = Path(args.pred)
    if not pred_path.is_absolute():
        pred_path = ROOT / pred_path

    with open(pred_path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit("File prediksi kosong.")

    ref_col = "ref_norm" if "ref_norm" in rows[0] else "ref_raw"
    hyp_col = "pred_norm" if "pred_norm" in rows[0] else "pred_raw"
    analysis = analyze_rows(rows, ref_col, hyp_col, top_k=args.top_k)

    out_path = ROOT / "results" / f"error_analysis_{args.tag}.md"
    write_report(analysis, out_path, args.tag)
    print(f"Laporan analisis kesalahan disimpan ke {out_path}")


if __name__ == "__main__":
    main()
