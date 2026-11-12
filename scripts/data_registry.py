"""
scripts/data_registry.py — versionnement des données et des modèles sans git-lfs ni DVC.

Principe : les binaires (checkpoints .pt, oracles .pkl, datasets .parquet) ne vont
pas dans git. Git ne versionne que `data/registry.json`, qui associe chaque fichier
à son empreinte SHA-256, sa taille, le commit git et la date. Les octets vivent sur
S3 (bucket `DATA_BUCKET` du .env), adressés par leur empreinte, donc immuables.

    venv/bin/python scripts/data_registry.py add data/models/xxx.pt            # enregistre (hash + méta)
    venv/bin/python scripts/data_registry.py push                               # envoie sur S3 ce qui manque
    venv/bin/python scripts/data_registry.py pull [chemin]                      # récupère depuis S3 (vérifie le hash)
    venv/bin/python scripts/data_registry.py status                             # fichiers modifiés / absents
    venv/bin/python scripts/data_registry.py scan                               # enregistre tous les artefacts connus

Chaque checkpoint produit par `neural_nets/train.py` contient déjà `dataset_sha256`
(empreinte du dataset d'entraînement) : le registre permet de retrouver exactement
le dataset d'un modèle.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REGISTRY_PATH = os.path.join(PROJECT_ROOT, "data", "registry.json")
DEFAULT_PATTERNS = [
    "data/models/*.pt", "data/models/legacy/*.pt",
    "data/models/**/*.pkl", "data/models/**/*.parquet",
    "data/storage/screened/super_dataset_*.parquet", "data/storage/screened/super_dataset_*_meta.json",
]


def sha256_of(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def load_registry() -> dict:
    if os.path.exists(REGISTRY_PATH):
        with open(REGISTRY_PATH) as f:
            return json.load(f)
    return {"version": 1, "files": {}}


def save_registry(reg: dict) -> None:
    os.makedirs(os.path.dirname(REGISTRY_PATH), exist_ok=True)
    with open(REGISTRY_PATH, "w") as f:
        json.dump(reg, f, indent=2, sort_keys=True)


def rel(path: str) -> str:
    return os.path.relpath(os.path.abspath(path), PROJECT_ROOT)


def add(paths: list[str], note: str | None = None) -> None:
    reg = load_registry()
    for p in paths:
        r = rel(p)
        if not os.path.isfile(p):
            print(f"⚠️ absent : {r}")
            continue
        digest = sha256_of(p)
        entry = reg["files"].get(r, {})
        if entry.get("sha256") == digest:
            print(f"= inchangé : {r}")
            continue
        reg["files"][r] = {
            "sha256": digest,
            "size": os.path.getsize(p),
            "git_commit": git_commit(),
            "registered": datetime.now().isoformat(timespec="seconds"),
            "note": note or entry.get("note", ""),
        }
        print(f"+ enregistré : {r} ({digest[:12]}…, {os.path.getsize(p)/1e6:.1f} MB)")
    save_registry(reg)


def scan(note: str | None = None) -> None:
    paths = []
    for pat in DEFAULT_PATTERNS:
        paths += glob.glob(os.path.join(PROJECT_ROOT, pat), recursive=True)
    add(sorted(set(paths)), note)


def status() -> int:
    reg = load_registry()
    bad = 0
    for r, e in sorted(reg["files"].items()):
        p = os.path.join(PROJECT_ROOT, r)
        if not os.path.exists(p):
            print(f"ABSENT    {r}"); bad += 1
        elif sha256_of(p) != e["sha256"]:
            print(f"MODIFIÉ   {r}"); bad += 1
        else:
            print(f"ok        {r}")
    return bad


def _s3():
    bucket = os.getenv("DATA_BUCKET")
    if not bucket:
        sys.exit("DATA_BUCKET absent du .env : indiquer le bucket S3 du registre.")
    import boto3
    from dotenv import load_dotenv
    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
    return boto3.client("s3", region_name=os.getenv("AWS_REGION")), bucket


def _key(entry: dict, r: str) -> str:
    return f"pairs_trading/{entry['sha256']}/{os.path.basename(r)}"


def push() -> None:
    s3, bucket = _s3()
    reg = load_registry()
    for r, e in sorted(reg["files"].items()):
        p = os.path.join(PROJECT_ROOT, r)
        if not os.path.exists(p):
            print(f"⚠️ absent localement : {r}"); continue
        key = _key(e, r)
        try:
            s3.head_object(Bucket=bucket, Key=key)
            print(f"= déjà sur S3 : {r}")
        except Exception:
            s3.upload_file(p, bucket, key)
            print(f"↑ envoyé : {r} -> s3://{bucket}/{key}")


def pull(paths: list[str] | None = None) -> None:
    s3, bucket = _s3()
    reg = load_registry()
    targets = [rel(p) for p in paths] if paths else list(reg["files"].keys())
    for r in targets:
        e = reg["files"].get(r)
        if e is None:
            print(f"⚠️ non enregistré : {r}"); continue
        p = os.path.join(PROJECT_ROOT, r)
        if os.path.exists(p) and sha256_of(p) == e["sha256"]:
            print(f"= à jour : {r}"); continue
        os.makedirs(os.path.dirname(p), exist_ok=True)
        s3.download_file(bucket, _key(e, r), p)
        ok = sha256_of(p) == e["sha256"]
        print(f"↓ {r} {'(hash vérifié)' if ok else '!! HASH DIFFÉRENT'}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add"); a.add_argument("paths", nargs="+"); a.add_argument("--note", default=None)
    s = sub.add_parser("scan"); s.add_argument("--note", default=None)
    sub.add_parser("status"); sub.add_parser("push")
    p = sub.add_parser("pull"); p.add_argument("paths", nargs="*")
    args = ap.parse_args()
    if args.cmd == "add":
        add(args.paths, args.note)
    elif args.cmd == "scan":
        scan(args.note)
    elif args.cmd == "status":
        sys.exit(1 if status() else 0)
    elif args.cmd == "push":
        push()
    elif args.cmd == "pull":
        pull(args.paths or None)


if __name__ == "__main__":
    main()
