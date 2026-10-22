"""Extract complete DINOv3 patch-wise embeddings for every SEM micrograph in manifest.parquet.

For each image we crop the databar -> 1920x1184 at full resolution (no resize distortion), then
optionally downscale to the variant's target size (see exp_common: DINO_VARIANT/DINO_DOWNSCALE
env vars), run DINOv3 ViT-S/16+, split the token sequence, and store three fp16 representations,
each addressable by `emb_index`:

  patch    (N, NUM_PATCHES, 384)  dense per-patch features -> reshape (GRID_H, GRID_W, 384)
  special  (N, S,    384)  special tokens: [:,0]=CLS/global, [:,1:]=register tokens (S usually 5)
  pooled   (N, 384)        mean over patch tokens: one image-level vector for retrieval

At the default DOWNSCALE=1.0: CROP=1920x1184, grid=120x74, NUM_PATCHES=8880 (the original run).

Storage is memory-mapped .npy (resumable): a `done` mask records completed indices so an
interrupted run resumes. Reads are trivial:  np.load("embeddings/patch.npy", mmap_mode="r").

Usage:
  DINO_VARIANT=ds50 DINO_DOWNSCALE=0.5 python embed_dinov3.py --dry-run --count 5
  DINO_VARIANT=ds50 DINO_DOWNSCALE=0.5 python embed_dinov3.py               # full run, resumable
  python embed_dinov3.py                                                    # original full-res run
"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

import exp_common as ec

ROOT = ec.ROOT
OUT_DIR = ec.EMB_DIR
MANIFEST = ec.MANIFEST
META = ec.EMB_META

MODEL_ID = "facebook/dinov3-vits16plus-pretrain-lvd1689m"
RAW_SIZE = ec.RAW_SIZE
BASE_CROP_W, BASE_CROP_H = ec.BASE_CROP_W, ec.BASE_CROP_H   # full-res databar-crop, always applied
CROP_W, CROP_H = ec.CROP_W, ec.CROP_H                       # final (possibly downscaled) size
GRID_H, GRID_W = ec.GRID_H, ec.GRID_W
NUM_PATCHES = ec.NUM_PATCHES
DOWNSCALE = ec.DOWNSCALE


def load_and_preprocess(path: Path, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Grayscale -> crop databar (full-res 1920x1184) -> optional downscale -> replicate to 3ch
    -> normalize. Returns (3, CROP_H, CROP_W) fp32."""
    img = Image.open(path).convert("L")
    if img.size != RAW_SIZE:
        img = img.resize(RAW_SIZE)  # guard for any oddly-sized frame
    img = img.crop((0, 0, BASE_CROP_W, BASE_CROP_H))   # (left, upper, right, lower) -> keep top rows
    if (CROP_W, CROP_H) != (BASE_CROP_W, BASE_CROP_H):
        img = img.resize((CROP_W, CROP_H), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0  # (H, W)
    arr = np.repeat(arr[None, :, :], 3, axis=0)      # (3, H, W)
    arr = (arr - mean[:, None, None]) / std[:, None, None]
    return arr


@torch.no_grad()
def embed_batch(model, device, batch_np):
    x = torch.from_numpy(np.stack(batch_np)).to(device, dtype=torch.float16, non_blocking=True)
    out = model(pixel_values=x).last_hidden_state           # (B, tokens, D)
    tokens = out.shape[1]
    num_special = tokens - NUM_PATCHES
    if num_special < 1:
        raise RuntimeError(f"unexpected token count {tokens} vs {NUM_PATCHES} patches (num_special={num_special})")
    special = out[:, :num_special, :]
    patch = out[:, num_special:, :]
    pooled = patch.mean(dim=1)
    return (special.to(torch.float16).cpu().numpy(),
            patch.to(torch.float16).cpu().numpy(),
            pooled.to(torch.float16).cpu().numpy(),
            num_special)


def open_or_create(path: Path, shape, dtype):
    if path.exists():
        arr = np.load(path, mmap_mode="r+")
        if arr.shape != tuple(shape) or arr.dtype != np.dtype(dtype):
            raise RuntimeError(f"{path} exists with shape/dtype {arr.shape}/{arr.dtype}, expected {shape}/{dtype}")
        return arr
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=tuple(shape))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=None, help="images to process from --start (default: all)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true", help="process --count images, print checks, write nothing")
    ap.add_argument("--overwrite", action="store_true", help="reprocess even if marked done")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    df = ec.load_manifest()
    N = len(df)
    paths = [ROOT / p for p in df["rel_path"].tolist()]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}  |  {torch.cuda.get_device_name(0) if device=='cuda' else 'CPU'}")
    print(f"variant: {ec.VARIANT or '(default, full-res)'}  downscale: {DOWNSCALE}  "
          f"crop: {CROP_W}x{CROP_H}  grid: {GRID_W}x{GRID_H}  patches: {NUM_PATCHES}")

    proc = AutoImageProcessor.from_pretrained(MODEL_ID)
    mean = np.array(proc.image_mean, dtype=np.float32)
    std = np.array(proc.image_std, dtype=np.float32)
    print(f"normalize mean={mean.tolist()} std={std.tolist()}")

    model = AutoModel.from_pretrained(MODEL_ID, dtype=torch.float16, attn_implementation="sdpa")
    model.eval().to(device)
    D = model.config.hidden_size
    print(f"model: {MODEL_ID}  hidden_size(D)={D}")

    # probe one forward to derive num_special (S)
    probe = load_and_preprocess(paths[args.start], mean, std)
    sp, pt, pl, S = embed_batch(model, device, [probe])
    print(f"probe: patch {pt.shape[1:]}  special {sp.shape[1:]}  pooled {pl.shape[1:]}  S={S}  "
          f"patch NaN={np.isnan(pt).any()}")
    assert pt.shape[1:] == (NUM_PATCHES, D), pt.shape

    end = N if args.count is None else min(N, args.start + args.count)
    idxs = list(range(args.start, end))

    if args.dry_run:
        print(f"[dry-run] would process {len(idxs)} images -> patch{(len(idxs),NUM_PATCHES,D)} "
              f"special{(len(idxs),S,D)} pooled{(len(idxs),D)} fp16; nothing written.")
        return

    patch_a = open_or_create(OUT_DIR / "patch.npy", (N, NUM_PATCHES, D), np.float16)
    special_a = open_or_create(OUT_DIR / "special.npy", (N, S, D), np.float16)
    pooled_a = open_or_create(OUT_DIR / "pooled.npy", (N, D), np.float16)
    done_a = open_or_create(OUT_DIR / "done.npy", (N,), np.bool_)

    META.write_text(json.dumps({
        "model_id": MODEL_ID, "N": N, "D": int(D), "num_special": int(S),
        "num_patches": NUM_PATCHES, "grid_h": GRID_H, "grid_w": GRID_W,
        "crop_wh": [CROP_W, CROP_H], "base_crop_wh": [BASE_CROP_W, BASE_CROP_H],
        "downscale": DOWNSCALE, "variant": ec.VARIANT or None, "dtype": "float16",
        "arrays": {"patch": [N, NUM_PATCHES, D], "special": [N, S, D], "pooled": [N, D]},
    }, indent=2))

    todo = [i for i in idxs if args.overwrite or not done_a[i]]
    print(f"processing {len(todo)} / {len(idxs)} images (skipping {len(idxs)-len(todo)} already done)")

    pool = ThreadPoolExecutor(max_workers=args.workers)
    t0 = time.time()
    n_done = 0
    for b0 in range(0, len(todo), args.batch):
        bidx = todo[b0:b0 + args.batch]
        batch_np = list(pool.map(lambda i: load_and_preprocess(paths[i], mean, std), bidx))
        sp, pt, pl, s = embed_batch(model, device, batch_np)
        assert s == S, f"num_special drifted: {s} != {S}"
        for k, i in enumerate(bidx):
            patch_a[i] = pt[k]
            special_a[i] = sp[k]
            pooled_a[i] = pl[k]
            done_a[i] = True
        n_done += len(bidx)
        if (b0 // args.batch) % 10 == 0 or b0 + args.batch >= len(todo):
            el = time.time() - t0
            ips = n_done / el if el else 0
            eta = (len(todo) - n_done) / ips if ips else 0
            print(f"  {n_done}/{len(todo)}  {ips:.2f} img/s  ETA {eta/60:.1f} min", flush=True)
            patch_a.flush(); special_a.flush(); pooled_a.flush(); done_a.flush()

    patch_a.flush(); special_a.flush(); pooled_a.flush(); done_a.flush()
    elapsed = time.time() - t0
    print(f"done. total marked complete: {int(done_a[:].sum())}/{N}  in {elapsed/60:.1f} min")

    if int(done_a[:].sum()) == N:
        meta = json.loads(META.read_text())
        meta["total_extraction_seconds"] = round(elapsed, 1)
        meta["extraction_img_per_sec"] = round(N / elapsed, 2) if elapsed else None
        META.write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
