"""Decode the base64-encoded checkpoint back to checkpoints/model.pt.

The weights are stored as base64 text (model.pt.b64) because GitHub's
file API used for this repo accepts UTF-8 text only, not binary files.
Run from this directory:

    python decode_weights.py

This writes checkpoints/model.pt (1.2 MB) next to model.pt.b64.
"""

import base64
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    src = HERE / "model.pt.b64"
    dst = HERE / "model.pt"
    if dst.exists():
        print(f"{dst} already exists — nothing to do.")
        return
    data = base64.b64decode(src.read_text())
    dst.write_bytes(data)
    print(f"wrote {dst} ({len(data):,} bytes)")


if __name__ == "__main__":
    main()
