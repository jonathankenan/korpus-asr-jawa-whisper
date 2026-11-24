#!/usr/bin/env bash
# Mengambil kedua korpus mentah ke folder lokal (di-gitignore) pada commit tertentu,
# supaya semua anggota memakai data yang sama persis.
set -e
cd "$(dirname "$0")/.."

K11_REPO=https://github.com/rlukassa/Korpus_Bahasa_Jawa.git
K05_REPO=https://github.com/angelinaefrina/Korpus_Bahasa_Jawa.git
K11_COMMIT=7f1c3efc10e03991b6594fdc80f46a63e50e2bec
K05_COMMIT=c4fc1e0d5a52a8bae7176b2eada08ced3774a9fc

fetch() {
  [ -d "$1" ] || git clone "$2" "$1"
  git -C "$1" fetch origin
  git -C "$1" checkout "$3"
}

fetch corpus_k11 "$K11_REPO" "$K11_COMMIT"
fetch corpus_k05 "$K05_REPO" "$K05_COMMIT"
echo "Selesai: corpus_k11 @ $K11_COMMIT, corpus_k05 @ $K05_COMMIT"
