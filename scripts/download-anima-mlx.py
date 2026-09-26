#!/usr/bin/env python3
"""Fetch pinned experimental Anima MLX weights, checking each Hub SHA-256."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
REPO = "xocialize/anima-mlx"
REV = "4cc65eea48a3b57373ccbe1467c13c38c68b1b3f"


def main():
    dest = ROOT / "models/xocialize-anima-mlx"
    dest.mkdir(parents=True, exist_ok=True)
    tree = json.loads(subprocess.check_output([
        "curl", "-fLsS", f"https://huggingface.co/api/models/{REPO}/tree/{REV}"]))
    manifest = {"repo": REPO, "revision": REV,
                "reference_commit": "2338e281ee533568db558963f6e3da064eb60ff0", "files": []}
    for item in tree:
        name = item["path"]
        if name == ".gitattributes":
            continue
        path = dest / name
        expected = item.get("lfs", {}).get("oid")
        def digest(p):
            with p.open("rb") as f:
                return hashlib.file_digest(f, "sha256").hexdigest()
        if not path.exists() or (expected and digest(path) != expected):
            part = path.with_suffix(path.suffix + ".part")
            subprocess.run(["curl", "-fL", "--retry", "3", "-C", "-", "-o", str(part),
                            f"https://huggingface.co/{REPO}/resolve/{REV}/{name}"], check=True)
            if expected and digest(part) != expected:
                raise RuntimeError(f"SHA-256 mismatch: {name}")
            part.rename(path)
        manifest["files"].append({"path": name, "bytes": path.stat().st_size, "sha256": digest(path)})
        print(f"verified {name}", flush=True)
    (ROOT / "benchmarks/anima-mlx-model.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
