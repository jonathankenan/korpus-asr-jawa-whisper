"""Evaluasi WER dan CER untuk hasil transkripsi ASR Bahasa Jawa.

Menghitung metrik utama sesuai SPEC.md bagian 5C:
  - Word Error Rate (WER)
  - Character Error Rate (CER)
  - Breakdown per penutur, dialek, status penutur, dan korpus (k11/k05)
  - Rata-rata & simpangan baku antar fold (sesuai catatan keterbatasan DATASET.md)
  - Tabel contoh prediksi (Audio | Referensi | Prediksi | WER | CER)

Contoh:
  python src/evaluate.py --pred results/predictions_fold1_pretrained.csv --tag pretrained
  python src/evaluate.py --pred-glob "results/predictions_fold*_pretrained.csv" --tag pretrained
  python src/evaluate.py --self-test
"""

import argparse
import collections
import csv
import glob
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

ROOT = Path(__file__).resolve().parent.parent


def edit_ops(ref: List[str], hyp: List[str]) -> Tuple[int, int, int, int]:
    """Hitung (substitutions, deletions, insertions, ref_len) dengan Levenshtein DP."""
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    op = [[0] * (m + 1) for _ in range(n + 1)]  # 0=match/sub, 1=del, 2=ins

    for i in range(1, n + 1):
        dp[i][0] = i
        op[i][0] = 1
    for j in range(1, m + 1):
        dp[0][j] = j
        op[0][j] = 2

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                op[i][j] = 0
            else:
                sub = dp[i - 1][j - 1] + 1
                dlt = dp[i - 1][j] + 1
                ins = dp[i][j - 1] + 1
                best = min(sub, dlt, ins)
                dp[i][j] = best
                op[i][j] = 0 if best == sub else (1 if best == dlt else 2)

    s = d = ins_c = 0
    i, j = n, m
    while i > 0 or j > 0:
        cur = op[i][j]
        if cur == 0:
            if ref[i - 1] != hyp[j - 1]:
                s += 1
            i -= 1
            j -= 1
        elif cur == 1:
            d += 1
            i -= 1
        else:
            ins_c += 1
            j -= 1
    return s, d, ins_c, n


def compute_corpus_metrics(rows: List[Dict[str, str]], ref_col: str, hyp_col: str) -> Dict[str, float]:
    """Hitung WER (%) dan CER (%) agregat untuk sekumpulan baris."""
    w_s = w_d = w_i = w_n = 0
    c_s = c_d = c_i = c_n = 0

    for r in rows:
        ref = (r.get(ref_col) or "").strip()
        hyp = (r.get(hyp_col) or "").strip()
        if not ref:
            continue

        s, d, i, n = edit_ops(ref.split(), hyp.split())
        w_s += s
        w_d += d
        w_i += i
        w_n += n

        cs, cd, ci, cn = edit_ops(list(ref), list(hyp))
        c_s += cs
        c_d += cd
        c_i += ci
        c_n += cn

    wer = 100.0 * (w_s + w_d + w_i) / max(1, w_n)
    cer = 100.0 * (c_s + c_d + c_i) / max(1, c_n)
    return {
        "n_utterances": len(rows),
        "ref_words": w_n,
        "ref_chars": c_n,
        "wer_pct": round(wer, 2),
        "cer_pct": round(cer, 2),
        "word_sub": w_s,
        "word_del": w_d,
        "word_ins": w_i,
    }


def utt_wer_cer(ref: str, hyp: str) -> Tuple[float, float]:
    s, d, i, n = edit_ops(ref.split(), hyp.split())
    cs, cd, ci, cn = edit_ops(list(ref), list(hyp))
    wer = 100.0 * (s + d + i) / max(1, n)
    cer = 100.0 * (cs + cd + ci) / max(1, cn)
    return round(wer, 2), round(cer, 2)


def mean_std(vals: List[float]) -> Tuple[float, float]:
    if not vals:
        return 0.0, 0.0
    mu = sum(vals) / len(vals)
    var = sum((v - mu) ** 2 for v in vals) / len(vals)
    return round(mu, 2), round(math.sqrt(var), 2)


def run_self_test():
    """Uji perhitungan WER dan CER pada kasus sintetis."""
    sample = [
        {"ref_norm": "aku mangan sego goreng", "hyp_norm": "aku mangan sego goreng"},
        {"ref_norm": "sugeng rawuh ing bandung", "hyp_norm": "sugeng rawuh nang bandung"},
    ]
    res = compute_corpus_metrics(sample, "ref_norm", "hyp_norm")
    assert res["wer_pct"] == 12.5, f"Expected 12.5% WER, got {res['wer_pct']}"
    assert res["word_sub"] == 1 and res["ref_words"] == 8
    print(f"Self-test lolos: WER={res['wer_pct']}%, CER={res['cer_pct']}%")


def main():
    parser = argparse.ArgumentParser(description="Evaluasi WER/CER ASR Bahasa Jawa")
    parser.add_argument("--config", default="configs/eval.yaml")
    parser.add_argument("--pred", default=None, help="Path satu file CSV prediksi")
    parser.add_argument("--pred-glob", default=None, help="Pola glob file prediksi lintas fold")
    parser.add_argument("--tag", default="eval", help="Tag untuk file output hasil")
    parser.add_argument("--self-test", action="store_true", help="Jalankan uji mandiri metrik")
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return

    with open(ROOT / args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["evaluation"]

    ref_col = cfg["ref_column"]
    hyp_col = cfg["hyp_column"]

    files: List[Path] = []
    if args.pred:
        files = [ROOT / args.pred if not Path(args.pred).is_absolute() else Path(args.pred)]
    elif args.pred_glob:
        files = [Path(p) for p in sorted(glob.glob(str(ROOT / args.pred_glob)))]
    else:
        raise SystemExit("Berikan --pred atau --pred-glob (atau --self-test).")

    if not files:
        raise SystemExit("Tidak ada file prediksi ditemukan.")

    all_rows: List[Dict[str, str]] = []
    per_fold = []
    for fp in files:
        with open(fp, encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        r_col = ref_col if ref_col in rows[0] else cfg["fallback_ref_column"]
        h_col = hyp_col if hyp_col in rows[0] else cfg["fallback_hyp_column"]
        metrics = compute_corpus_metrics(rows, r_col, h_col)
        per_fold.append({"file": fp.name, **metrics})
        for r in rows:
            r["_ref"] = r.get(r_col, "")
            r["_hyp"] = r.get(h_col, "")
        all_rows.extend(rows)

    overall = compute_corpus_metrics(all_rows, "_ref", "_hyp")
    wer_mean, wer_std = mean_std([f["wer_pct"] for f in per_fold])
    cer_mean, cer_std = mean_std([f["cer_pct"] for f in per_fold])

    out_dir = ROOT / cfg["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "tag": args.tag,
        "overall": overall,
        "fold_mean_wer": wer_mean,
        "fold_std_wer": wer_std,
        "fold_mean_cer": cer_mean,
        "fold_std_cer": cer_std,
        "per_fold": per_fold,
    }
    summary_path = out_dir / f"eval_summary_{args.tag}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Breakdown per atribut (speaker, dialect, speaker_status, group)
    breakdown_rows = []
    for attr in cfg["group_by"]:
        grouped = collections.defaultdict(list)
        for r in all_rows:
            grouped[r.get(attr, "") or "unknown"].append(r)
        for val, grp_rows in sorted(grouped.items()):
            m = compute_corpus_metrics(grp_rows, "_ref", "_hyp")
            breakdown_rows.append({"attribute": attr, "value": val, **m})

    breakdown_path = out_dir / f"eval_breakdown_{args.tag}.csv"
    if breakdown_rows:
        with open(breakdown_path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(breakdown_rows[0].keys()))
            w.writeheader()
            w.writerows(breakdown_rows)

    # Contoh prediksi Markdown (SPEC.md 5C)
    examples_path = out_dir / f"eval_examples_{args.tag}.md"
    n_ex = cfg.get("num_examples", 10)
    with open(examples_path, "w", encoding="utf-8") as f:
        f.write(f"# Contoh Hasil Pengenalan Ucapan ({args.tag})\n\n")
        f.write("| Audio / Ujaran | Transkripsi Referensi | Prediksi Model | WER (%) | CER (%) |\n")
        f.write("|---|---|---|---|---|\n")
        for r in all_rows[:n_ex]:
            w_val, c_val = utt_wer_cer(r["_ref"], r["_hyp"])
            ref_txt = r["_ref"].replace("|", "\\|")
            hyp_txt = r["_hyp"].replace("|", "\\|")
            f.write(f"| {r.get('utt_id', '-')} | {ref_txt} | {hyp_txt} | {w_val} | {c_val} |\n")

    print(f"Evaluasi ({args.tag}) selesai:")
    print(f"  Overall : WER={overall['wer_pct']}% | CER={overall['cer_pct']}% ({overall['n_utterances']} ujaran)")
    if len(per_fold) > 1:
        print(f"  Antar-fold: WER={wer_mean}% ± {wer_std}% | CER={cer_mean}% ± {cer_std}%")
    print(f"  Output  : {summary_path.name}, {breakdown_path.name}, {examples_path.name}")


if __name__ == "__main__":
    main()
