"""Unduh bobot Whisper pada revisi yang dikunci di configs/models.yaml ke cache Hugging Face.

Contoh:
  python scripts/download_models.py
  python scripts/download_models.py openai/whisper-tiny
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")  # model publik, tidak perlu token

import yaml
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent.parent
PATTERNS = ["*.json", "*.txt", "model.safetensors"]


def main():
    with open(ROOT / "configs" / "models.yaml", encoding="utf-8") as f:
        models = yaml.safe_load(f)["models"]
    names = sys.argv[1:] or list(models)
    for name in names:
        rev = models[name]["revision"]
        print(f"Mengunduh {name}@{rev[:10]} ...", flush=True)
        path = snapshot_download(name, revision=rev, allow_patterns=PATTERNS)
        size = sum(p.stat().st_size for p in Path(path).iterdir() if p.is_file())
        print(f"  selesai: {size / 1e6:.0f} MB di {path}", flush=True)


if __name__ == "__main__":
    main()
