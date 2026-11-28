"""Evaluasi WER dan CER hasil transkripsi ASR Bahasa Jawa (spesifikasi Fase 2 §5C).

Prediksi dibaca dari results/predictions/<exp>/fold<k>.csv. Referensi dan atribut diambil
dari data/manifest_normalized.csv berdasarkan utt_id saat evaluasi dijalankan, sehingga
koreksi transkripsi cukup diikuti dengan evaluasi ulang tanpa inferensi ulang.
Referensi dan prediksi dinormalisasi dengan fungsi yang sama (src/normalize_text.py).

Keluaran:
  results/eval/<exp>/summary.json     metrik keseluruhan, per korpus, per fold
  results/eval/<exp>/breakdown.csv    per penutur, dialek, status, kejadian, durasi, bahasa
  results/eval/<exp>/utterances.csv   skor per ujaran (tanpa teks)
  results/eval/<exp>/examples.md      contoh referensi vs prediksi
  results/summary.md, summary.csv     perbandingan semua eksperimen

Contoh:
  python src/evaluate.py              # semua eksperimen di results/predictions
  python src/evaluate.py --self-test
"""

import argparse
import collections
import csv
import json
import math
from datetime import date
from pathlib import Path
from typing import Dict, List, Tuple

from data import (FOLDS, clean_exclusion_reasons, dialect_family, duration_bucket,
                  event_categories, load_manifest, speaker_table, test_fold_by_speaker)
from normalize_text import normalize

ROOT = Path(__file__).resolve().parent.parent
PRED_DIR = ROOT / "results" / "predictions"
EVAL_DIR = ROOT / "results" / "eval"
TRAIN_DIR = ROOT / "results" / "training"
GROUP_NAMES = {"k05": "kelompok 5", "k11": "kelompok ini (draf ASR)"}
# Penutur untuk contoh lintas model: campuran korpus, dialek, dan status penutur
EXAMPLE_SPEAKERS = ["S11", "S13", "S18", "S03", "S07", "S10"]
COUNT_KEYS = ("ws", "wd", "wi", "wn", "cs", "cd", "ci", "cn")


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


def score(ref: str, hyp: str) -> Dict[str, int]:
    """Hitungan kesalahan kata (w*) dan karakter (c*, spasi ikut dihitung) satu ujaran."""
    ws, wd, wi, wn = edit_ops(ref.split(), hyp.split())
    cs, cd, ci, cn = edit_ops(list(ref), list(hyp))
    return {"ws": ws, "wd": wd, "wi": wi, "wn": wn, "cs": cs, "cd": cd, "ci": ci, "cn": cn}


def aggregate(rows: List[Dict]) -> Dict[str, float]:
    """WER/CER agregat (jumlah kesalahan dibagi jumlah kata/karakter referensi)."""
    total = {k: sum(r[k] for r in rows) for k in COUNT_KEYS}
    wer = 100.0 * (total["ws"] + total["wd"] + total["wi"]) / max(1, total["wn"])
    cer = 100.0 * (total["cs"] + total["cd"] + total["ci"]) / max(1, total["cn"])
    return {
        "n_utterances": len(rows),
        "ref_words": total["wn"],
        "ref_chars": total["cn"],
        "wer_pct": round(wer, 2),
        "cer_pct": round(cer, 2),
        "word_sub": total["ws"],
        "word_del": total["wd"],
        "word_ins": total["wi"],
    }


def compute_corpus_metrics(rows: List[Dict[str, str]], ref_col: str, hyp_col: str) -> Dict[str, float]:
    """Hitung WER (%) dan CER (%) agregat dari kolom referensi dan prediksi."""
    scored = []
    for r in rows:
        ref = (r.get(ref_col) or "").strip()
        if ref:
            scored.append(score(ref, (r.get(hyp_col) or "").strip()))
    return aggregate(scored)


def mean_std(vals: List[float]) -> Tuple[float, float]:
    if not vals:
        return 0.0, 0.0
    mu = sum(vals) / len(vals)
    var = sum((v - mu) ** 2 for v in vals) / len(vals)
    return round(mu, 2), round(math.sqrt(var), 2)


def load_predictions(exp: str) -> List[Dict]:
    """Gabungkan prediksi test satu eksperimen dengan referensi dan atribut manifest."""
    manifest = {r["utt_id"]: r for r in load_manifest()}
    speakers = speaker_table()
    rows = []
    for path in sorted((PRED_DIR / exp).glob("fold*.csv")):
        with open(path, encoding="utf-8", newline="") as f:
            for p in csv.DictReader(f):
                m = manifest.get(p["utt_id"])
                if m is None or p["split"] != "test":
                    continue
                ref, hyp = normalize(m["text_raw"]), normalize(p["pred_raw"])
                if not ref:
                    continue
                spk = speakers[m["speaker"]]
                rows.append({
                    "utt_id": p["utt_id"], "speaker": m["speaker"], "group": m["group"],
                    "fold": int(p["fold"]), "ref": ref, "hyp": hyp, "gender": m["gender"],
                    "speaker_status": m["speaker_status"],
                    "dialect_family": dialect_family(spk["dialect"]),
                    "language_label": m.get("language_label") or "tidak dilabeli",
                    "events": event_categories(m.get("event", "")),
                    "duration": float(m["duration"]),
                    "duration_bucket": duration_bucket(float(m["duration"])),
                    "excluded": clean_exclusion_reasons(m),
                    **score(ref, hyp),
                })
    rows.sort(key=lambda r: r["utt_id"])
    return rows


def expected_per_fold() -> Dict[int, int]:
    fold_of = test_fold_by_speaker()
    counts = collections.Counter(
        fold_of[r["speaker"]] for r in load_manifest() if normalize(r["text_raw"])
    )
    return {k: counts[k] for k in FOLDS}


def run_info(exp: str) -> Dict:
    """Model, decoding, dan waktu inferensi dari berkas fold<k>.json."""
    info = {"models": [], "num_beams": None, "peak_dbfs": None, "audio_seconds": 0.0,
            "wall_seconds": 0.0, "device": set()}
    for path in sorted((PRED_DIR / exp).glob("fold*.json")):
        meta = json.loads(path.read_text(encoding="utf-8"))
        sig = meta["signature"]
        info["models"].append(sig["model"])
        info["num_beams"], info["peak_dbfs"] = sig["num_beams"], sig["peak_dbfs"]
        info["audio_seconds"] += meta["audio_seconds"]
        info["wall_seconds"] += meta["wall_seconds"]
        info["device"].add(meta["device"])
    info["device"] = ", ".join(sorted(info["device"]))
    info["rtf"] = round(info["wall_seconds"] / info["audio_seconds"], 3) if info["audio_seconds"] else None
    return info


def training_records(exp: str) -> List[Dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted((TRAIN_DIR / exp).glob("fold*.json"))]


def pick_examples(rows: List[Dict], per_group: int = 5) -> List[Dict]:
    """Contoh terbaik, menengah, dan buruk dari penutur berbeda pada tiap korpus."""
    chosen = []
    for group in ("k05", "k11"):
        pool = [r for r in rows if r["group"] == group and r["wn"] >= 4]
        pool.sort(key=lambda r: ((r["ws"] + r["wd"] + r["wi"]) / r["wn"], r["utt_id"]))
        if not pool:
            continue
        used = set()
        for q in [i / (per_group - 1) * 0.95 for i in range(per_group)]:
            start = int(q * (len(pool) - 1))
            for r in pool[start:] + pool[:start]:
                if r["speaker"] not in used:
                    chosen.append(r)
                    used.add(r["speaker"])
                    break
    return chosen


def md_cell(text: str) -> str:
    return text.replace("|", "\\|") or "(kosong)"


def pct(part: int, whole: int) -> float:
    return round(100.0 * part / max(1, whole), 2)


def evaluate_experiment(exp: str) -> Dict:
    rows = load_predictions(exp)
    if not rows:
        return {}
    expected = expected_per_fold()
    got = collections.Counter(r["fold"] for r in rows)
    complete = [k for k in FOLDS if got[k] == expected[k]]
    partial = {k: f"{got[k]}/{expected[k]}" for k in FOLDS if 0 < got[k] < expected[k]}
    clean = [r for r in rows if not r["excluded"]]

    def by_group(subset):
        out = {"semua": aggregate(subset)}
        for g in ("k05", "k11"):
            sel = [r for r in subset if r["group"] == g]
            if sel:
                out[g] = aggregate(sel)
        return out

    per_fold = {k: by_group([r for r in rows if r["fold"] == k]) for k in complete}
    fold_stats = {}
    for g in ("semua", "k05", "k11"):
        for metric in ("wer_pct", "cer_pct"):
            vals = [per_fold[k][g][metric] for k in complete if g in per_fold[k]]
            fold_stats[f"{g}_{metric}"] = mean_std(vals)

    summary = {
        "exp": exp,
        "complete_folds": complete,
        "partial_folds": partial,
        "all_folds_complete": complete == FOLDS,
        "metrics": by_group(rows),
        "metrics_clean_subset": by_group(clean) if clean else {},
        "per_fold": per_fold,
        "fold_mean_std": fold_stats,
        "empty_predictions": sum(1 for r in rows if not r["hyp"]),
        "insertions_exceed_reference": sum(1 for r in rows if r["wi"] > r["wn"]),
        "run": run_info(exp),
    }
    out = EVAL_DIR / exp
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # Rincian per atribut
    breakdown = []
    attrs = ["group", "speaker", "gender", "speaker_status", "dialect_family",
             "duration_bucket", "language_label", "events"]
    for attr in attrs:
        grouped = collections.defaultdict(list)
        for r in rows:
            if attr == "language_label" and r["group"] != "k11":
                continue  # hanya korpus kelompok ini yang punya label bahasa
            values = r[attr] if attr == "events" else [r[attr]]
            for v in values:
                grouped[(r["group"] if attr == "events" else "", v)].append(r)
        for (group, value), subset in sorted(grouped.items()):
            label = f"{group}: {value}" if group else value
            breakdown.append({"attribute": attr, "value": label, **aggregate(subset)})
    for reason, count in sorted(collections.Counter(x for r in rows for x in r["excluded"]).items()):
        subset = [r for r in rows if reason in r["excluded"]]
        breakdown.append({"attribute": "dikeluarkan_dari_subset_bersih", "value": reason, **aggregate(subset)})
    with open(out / "breakdown.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(breakdown[0].keys()))
        w.writeheader()
        w.writerows(breakdown)

    with open(out / "utterances.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["utt_id", "speaker", "group", "fold", "wer_pct", "cer_pct", "sub", "del", "ins", "ref_words"])
        for r in rows:
            w.writerow([r["utt_id"], r["speaker"], r["group"], r["fold"],
                        pct(r["ws"] + r["wd"] + r["wi"], r["wn"]),
                        pct(r["cs"] + r["cd"] + r["ci"], r["cn"]),
                        r["ws"], r["wd"], r["wi"], r["wn"]])

    lines = [f"# Contoh Hasil Pengenalan Ucapan ({exp})", "",
             "Dipilih otomatis: urutan WER dari terbaik ke buruk, satu per penutur, ujaran ≥ 4 kata.",
             "Referensi korpus k11 masih draf ASR.", "",
             "| Audio / Ujaran | Transkripsi Referensi | Prediksi Model | WER (%) | CER (%) |",
             "|---|---|---|---|---|"]
    for r in pick_examples(rows):
        lines.append(f"| {r['utt_id']} ({r['group']}) | {md_cell(r['ref'])} | {md_cell(r['hyp'])} | "
                     f"{pct(r['ws'] + r['wd'] + r['wi'], r['wn'])} | {pct(r['cs'] + r['cd'] + r['ci'], r['cn'])} |")
    (out / "examples.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {**summary, "rows": rows}


def table(header: List[str], body: List[List[str]]) -> List[str]:
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + [
        "| " + " | ".join(str(c) for c in row) + " |" for row in body] + [""]


def cell(metrics: Dict, group: str) -> str:
    m = metrics.get(group)
    return f"{m['wer_pct']} / {m['cer_pct']}" if m else "-"


def write_summary(results: Dict[str, Dict]) -> None:
    """Ringkasan perbandingan seluruh eksperimen."""
    full = [e for e, r in results.items() if r["all_folds_complete"]]
    other = [e for e in results if e not in full]
    full.sort(key=lambda e: -results[e]["metrics"]["semua"]["wer_pct"])
    lines = ["# Ringkasan Hasil Eksperimen", "",
             f"Dibuat oleh `src/evaluate.py` pada {date.today().isoformat()}. Angka adalah **WER / CER (%)**, "
             "makin kecil makin baik.", "",
             "> Transkripsi korpus kelompok ini (k11) masih **draf ASR yang belum dikoreksi manusia**, "
             "sehingga skor k11 tidak dapat dianggap akurasi sebenarnya. Pakai kolom k05 sebagai hasil "
             "utama (status review k05 juga tidak terdokumentasi).", ""]

    lines += ["## 1. Eksperimen lengkap (5 fold)", ""]
    if full:
        body = []
        for e in full:
            r = results[e]
            mu, sd = r["fold_mean_std"]["semua_wer_pct"]
            mu5, sd5 = r["fold_mean_std"]["k05_wer_pct"]
            body.append([e, r["run"]["num_beams"], cell(r["metrics"], "k05"), cell(r["metrics"], "k11"),
                         cell(r["metrics"], "semua"), f"{mu5} ± {sd5}", f"{mu} ± {sd}", r["run"]["rtf"]])
        lines += table(["Eksperimen", "Beam", "k05", "k11 (draf)", "Gabungan",
                        "WER k05 rata-rata fold ± SD", "WER gabungan rata-rata fold ± SD", "RTF*"], body)
        lines += ["*RTF = waktu inferensi dibagi durasi audio; bergantung perangkat (lihat summary.json).", ""]

        lines += ["## 2. Subset bersih (pembanding)", "",
                  "Tanpa segmen: pertanyaan panduan, hening/halusinasi, berlabel bukan Jawa, referensi "
                  "ber-tag atau berangka, durasi < 1 s, atau laju teks > 25 karakter/s.", ""]
        body = []
        for e in full:
            m = results[e]["metrics_clean_subset"]
            body.append([e, m["semua"]["n_utterances"], cell(m, "k05"), cell(m, "k11"), cell(m, "semua")])
        lines += table(["Eksperimen", "Ujaran", "k05", "k11 (draf)", "Gabungan"], body)

        lines += ["## 3. WER (%) per fold (gabungan)", ""]
        lines += table(["Eksperimen"] + [f"Fold {k}" for k in FOLDS],
                       [[e] + [results[e]["per_fold"][k]["semua"]["wer_pct"] for k in FOLDS] for e in full])

        for number, (title, attr) in enumerate([("penutur", "speaker"), ("kelompok dialek", "dialect_family"),
                                                ("status penutur", "speaker_status"), ("gender", "gender"),
                                                ("durasi ujaran", "duration_bucket")], start=4):
            values = sorted({r[attr] for r in results[full[0]]["rows"]})
            body = []
            for v in values:
                row = [v, len([r for r in results[full[0]]["rows"] if r[attr] == v])]
                for e in full:
                    row.append(aggregate([r for r in results[e]["rows"] if r[attr] == v])["wer_pct"])
                body.append(row)
            lines += [f"## {number}. WER (%) per {title}", ""]
            lines += table([title.capitalize(), "Ujaran"] + full, body)
    else:
        lines += ["Belum ada eksperimen yang lengkap 5 fold.", ""]

    lines += ["## 9. Perbandingan pada fold 1 (termasuk pencarian learning rate)", "",
              "Learning rate dipilih dari WER **validation**; kolom test hanya dilaporkan.", ""]
    body = []
    for e, r in sorted(results.items()):
        if 1 not in r["complete_folds"]:
            continue
        rec = next((x for x in training_records(e) if x["fold"] == 1), None)
        body.append([e, rec["settings"]["learning_rate"] if rec else "-",
                     f"{rec['val_before']['wer']}" if rec else "-",
                     f"{rec['val_best']['wer']}" if rec else "-",
                     cell(r["per_fold"][1], "k05"), cell(r["per_fold"][1], "semua")])
    lines += table(["Eksperimen", "LR", "WER val sebelum", "WER val terbaik", "Test k05", "Test gabungan"], body)

    records = [x for e in results for x in training_records(e)]
    if records:
        lines += ["## 10. Catatan pelatihan", ""]
        lines += table(["Eksperimen", "Fold", "Encoder", "Parameter dilatih", "LR", "Epoch", "Data latih",
                        "WER val sebelum → terbaik", "Menit", "Perangkat"],
                       [[x["exp"], x["fold"], "beku" if x["settings"]["freeze_encoder"] else "dilatih",
                         f"{x['trainable_parameters'] / 1e6:.1f}M", x["settings"]["learning_rate"],
                         x["settings"]["epochs"], x["n_train"],
                         f"{x['val_before']['wer']} → {x['val_best']['wer']}",
                         round(x["wall_seconds"] / 60), x["device"]]
                        for x in sorted(records, key=lambda x: (x["exp"], x["fold"]))])

    if full:
        lines += ["## 11. Contoh prediksi lintas model", ""]
        ref_rows = results[full[0]]["rows"]
        for spk in EXAMPLE_SPEAKERS:
            pick = next((r for r in ref_rows if r["speaker"] == spk and 6 <= r["wn"] <= 14
                         and not r["excluded"]), None)
            if not pick:
                continue
            lines += [f"**{pick['utt_id']}** ({GROUP_NAMES[pick['group']]})", ""]
            body = [["Referensi", md_cell(pick["ref"]), "", ""]]
            for e in full:
                r = next(x for x in results[e]["rows"] if x["utt_id"] == pick["utt_id"])
                body.append([e, md_cell(r["hyp"]), pct(r["ws"] + r["wd"] + r["wi"], r["wn"]),
                             pct(r["cs"] + r["cd"] + r["ci"], r["cn"])])
            lines += table(["Sumber", "Teks", "WER (%)", "CER (%)"], body)

    if other:
        lines += ["## 12. Eksperimen belum lengkap", ""]
        lines += table(["Eksperimen", "Fold lengkap", "Fold sebagian", "WER / CER (yang sudah ada)"],
                       [[e, results[e]["complete_folds"] or "-", results[e]["partial_folds"] or "-",
                         cell(results[e]["metrics"], "semua")] for e in sorted(other)])

    (ROOT / "results" / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with open(ROOT / "results" / "summary.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["exp", "complete_folds", "group", "n_utterances", "wer_pct", "cer_pct",
                    "clean_n", "clean_wer_pct", "clean_cer_pct"])
        for e, r in sorted(results.items()):
            for g, m in r["metrics"].items():
                c = r["metrics_clean_subset"].get(g, {})
                w.writerow([e, " ".join(map(str, r["complete_folds"])), g, m["n_utterances"], m["wer_pct"],
                            m["cer_pct"], c.get("n_utterances", ""), c.get("wer_pct", ""), c.get("cer_pct", "")])


def run_self_test():
    """Uji perhitungan WER dan CER pada kasus sintetis."""
    sample = [
        {"ref_norm": "aku mangan sego goreng", "hyp_norm": "aku mangan sego goreng"},
        {"ref_norm": "sugeng rawuh ing bandung", "hyp_norm": "sugeng rawuh nang bandung"},
    ]
    res = compute_corpus_metrics(sample, "ref_norm", "hyp_norm")
    assert res["wer_pct"] == 12.5, f"Expected 12.5% WER, got {res['wer_pct']}"
    assert res["word_sub"] == 1 and res["ref_words"] == 8
    s = score("abc de", "abd")
    assert (s["ws"], s["wd"], s["wn"]) == (1, 1, 2) and s["cn"] == 6
    print(f"Self-test lolos: WER={res['wer_pct']}%, CER={res['cer_pct']}%")


def main():
    parser = argparse.ArgumentParser(description="Evaluasi WER/CER ASR Bahasa Jawa")
    parser.add_argument("--self-test", action="store_true", help="Jalankan uji mandiri metrik")
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return

    experiments = sorted(p.name for p in PRED_DIR.glob("*") if p.is_dir()) if PRED_DIR.exists() else []
    results = {}
    for exp in experiments:
        res = evaluate_experiment(exp)
        if res:
            results[exp] = res
    if not results:
        raise SystemExit("Belum ada prediksi di results/predictions.")
    write_summary(results)
    for exp, r in results.items():
        status = "lengkap" if r["all_folds_complete"] else f"fold lengkap {r['complete_folds']}"
        print(f"  {exp:24s} {status:22s} k05 {cell(r['metrics'], 'k05'):16s} "
              f"gabungan {cell(r['metrics'], 'semua')}  (WER / CER %)")
    print("Ringkasan: results/summary.md")


if __name__ == "__main__":
    main()
