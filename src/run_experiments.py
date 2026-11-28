"""Jalankan skenario eksperimen pada configs/experiments.yaml: latih, inferensi, evaluasi, analisis.

Tiap langkah dapat dilanjutkan: fold yang sudah dilatih dan segmen yang sudah
ditranskripsi dilewati. Jika proses terputus, ulangi perintah yang sama.

Contoh:
  python src/run_experiments.py --group local
  python src/run_experiments.py small-pretrained
  python src/run_experiments.py --list
"""

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import yaml

from evaluate import expected_per_fold

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"


def run(script: str, *args) -> None:
    cmd = [sys.executable, "-u", str(SRC / script)] + [str(a) for a in args]
    print(f"\n$ {script} {' '.join(str(a) for a in args)}", flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def log(message: str) -> None:
    print(f"\n===== {datetime.now():%Y-%m-%d %H:%M:%S} | {message} =====", flush=True)


def keep_awake() -> None:
    """Minta Windows tidak tidur selama proses ini hidup (tidak mengubah pengaturan daya)."""
    if sys.platform == "win32":
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
        print("Mode tidur otomatis ditahan selama runner berjalan (menutup layar tetap menidurkan).")


def best_source(sources: List[str]) -> Dict:
    """Catatan latih (fold 1) dengan WER validation terkecil di antara eksperimen sumber."""
    records = []
    for name in sources:
        path = ROOT / "results" / "training" / name / "fold1.json"
        if not path.exists():
            raise SystemExit(f"lr_from butuh {name} selesai lebih dulu ({path} belum ada).")
        records.append(json.loads(path.read_text(encoding="utf-8")))
    best = min(records, key=lambda r: (r["val_best"]["wer"], r["val_best"]["cer"]))
    for r in records:
        print(f"  {r['exp']}: lr {r['settings']['learning_rate']} -> WER val {r['val_best']['wer']}%")
    print(f"  dipilih: {best['exp']} (lr {best['settings']['learning_rate']})", flush=True)
    return best


def fold_done(name: str, fold: int, exp: Dict, lr) -> bool:
    """Fold fine-tuning dianggap selesai jika catatan latihnya cocok dan prediksinya lengkap.

    Berlaku juga bila bobot model tidak disimpan lagi (mis. sesi Colab terputus).
    """
    record = ROOT / "results" / "training" / name / f"fold{fold}.json"
    preds = ROOT / "results" / "predictions" / name / f"fold{fold}.csv"
    if not record.exists() or not preds.exists():
        return False
    s = json.loads(record.read_text(encoding="utf-8"))["settings"]
    same = (s["model"] == exp["model"] and s["learning_rate"] == float(lr)
            and s["epochs"] == exp.get("epochs", s["epochs"])
            and s["freeze_encoder"] == bool(exp.get("freeze_encoder")))
    with open(preds, encoding="utf-8", newline="") as f:
        n = sum(1 for _ in csv.DictReader(f))
    return same and n >= expected_per_fold()[fold]


def infer_args(exp: Dict) -> List:
    args = []
    if "num_beams" in exp:
        args += ["--num-beams", exp["num_beams"]]
    if "peak_dbfs" in exp:
        args += ["--peak-dbfs", exp["peak_dbfs"]]
    if "limit" in exp:  # hanya untuk uji cepat
        args += ["--limit", exp["limit"]]
    return args


def run_experiment(name: str, exps: Dict[str, Dict]) -> None:
    exp = exps[name]
    log(f"{name} — {exp['purpose']}")
    t0 = time.time()
    if exp["kind"] == "pretrained":
        run("infer.py", "--exp", name, "--model", exp["model"], "--fold", "all", *infer_args(exp))
    else:
        source, lr = None, exp.get("lr")
        if "lr_from" in exp:
            source = best_source(exp["lr_from"])
            lr = source["settings"]["learning_rate"]
        for fold in exp["folds"]:
            if fold_done(name, fold, exp, lr):
                print(f"  fold {fold}: sudah dilatih dan diprediksi, dilewati", flush=True)
                continue
            reuse = source is not None and fold == source["fold"]
            if reuse:
                # Konfigurasi identik dengan run pencarian LR: pakai model yang sama.
                model_dir = f"models/{source['exp']}/fold{fold}"
                record = ROOT / "results" / "training" / name / f"fold{fold}.json"
                record.parent.mkdir(parents=True, exist_ok=True)
                record.write_text(json.dumps({**source, "exp": name, "reused_from": source["exp"]},
                                             indent=2), encoding="utf-8")
                print(f"  fold {fold}: memakai ulang model {model_dir}", flush=True)
            else:
                model_dir = f"models/{name}/fold{fold}"
                train = ["--exp", name, "--model", exp["model"], "--fold", fold, "--lr", lr]
                for key, flag in (("epochs", "--epochs"), ("grad_accum", "--grad-accum"),
                                  ("warmup_steps", "--warmup-steps"), ("batch_size", "--batch-size"),
                                  ("max_train", "--max-train"), ("max_val", "--max-val")):
                    if key in exp:
                        train += [flag, exp[key]]
                if exp.get("freeze_encoder"):
                    train.append("--freeze-encoder")
                run("train.py", *train)
            run("infer.py", "--exp", name, "--model", model_dir, "--fold", fold, *infer_args(exp))

    analysis = ["--exp", name]
    baseline = exp.get("baseline")
    if baseline and (ROOT / "results" / "predictions" / baseline).exists():
        analysis += ["--baseline", baseline]
    run("evaluate.py")
    run("analyze_errors.py", *analysis)
    log(f"{name} selesai dalam {(time.time() - t0) / 60:.1f} menit")


def main():
    parser = argparse.ArgumentParser(description="Runner eksperimen ASR Bahasa Jawa")
    parser.add_argument("names", nargs="*", help="Nama eksperimen")
    parser.add_argument("--group", choices=["local", "gpu"], help="Jalankan satu grup")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--keep-awake", action="store_true", help="Tahan tidur otomatis (Windows)")
    args = parser.parse_args()

    with open(ROOT / "configs" / "experiments.yaml", encoding="utf-8") as f:
        exps = yaml.safe_load(f)["experiments"]

    if args.list:
        for name, e in exps.items():
            print(f"  {name:24s} {e['group']:6s} {e['kind']:10s} {e['purpose']}")
        return

    names = args.names or [n for n, e in exps.items() if e["group"] == args.group]
    if not names:
        raise SystemExit("Sebutkan nama eksperimen atau --group.")
    unknown = [n for n in names if n not in exps]
    if unknown:
        raise SystemExit(f"Eksperimen tidak dikenal: {unknown}")
    if args.keep_awake:
        keep_awake()
    failed = []
    for name in names:
        try:
            run_experiment(name, exps)
        except (subprocess.CalledProcessError, SystemExit) as error:
            failed.append(name)
            log(f"{name} GAGAL: {error}")
    if failed:
        raise SystemExit(f"Eksperimen gagal: {failed}. Perbaiki lalu ulangi perintah yang sama.")
    log("semua eksperimen yang diminta selesai")


if __name__ == "__main__":
    main()
