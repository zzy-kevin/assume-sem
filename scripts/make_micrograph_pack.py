#!/usr/bin/env python3
"""Build the downsampled micrograph pack shipped with the repo.

Writes the 1,543 evaluation micrographs (data/real/eval_manifest.parquet) to
data/micrographs/<class>/<image_id>.jpg at 384x288 (full frame, databar kept so the
usual top-75% crop still applies). Source: Aversa et al. 2018 SEM dataset, CC BY 4.0.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = PROJECT_ROOT / "data" / "real" / "eval_manifest.parquet"
OUT = PROJECT_ROOT / "data" / "micrographs"
SIZE = (384, 288)
QUALITY = 82


def main():
    df = pd.read_parquet(MANIFEST)
    for row in df.itertuples():
        dst = OUT / row.rel_path
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(row.full_path) as im:
            im.convert("L").resize(SIZE, Image.Resampling.LANCZOS).save(dst, quality=QUALITY, optimize=True)
    print(f"wrote {len(df)} micrographs to {OUT}")


if __name__ == "__main__":
    main()
