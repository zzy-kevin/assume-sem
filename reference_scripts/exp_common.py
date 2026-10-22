"""Shared infrastructure for the Part-2 structure<->property experiment suite.

- load_manifest() / synthesis_feature_cols() / sample_table(): metadata + the 51 informative
  synthesis columns (57 candidate cols A-BE minus 6 zero-variance ones, detected automatically).
- load_pooled() / load_patch(): mmap the embedding arrays produced by embed_dinov3.py.
- sample_fold_map() / image_fold_array(): ONE fixed 5-fold split over the 298 samples (seed=42),
  reused everywhere so image-level and sample-level experiments are leakage-free and comparable.
- cv_predict_scalar(): generic scalar-target 5-fold CV loop (impute -> scale -> optional PCA -> fit).
- regression_metrics() / multioutput_metrics(): standard metric dicts.
- log_result(): append an aggregated (mean+/-std) row to results/metrics.csv and write a per-run JSON.

Resolution variants: set env vars DINO_VARIANT (directory suffix, e.g. "ds50") and DINO_DOWNSCALE
(float, e.g. "0.5") before running ANY script (embed_dinov3.py, phase_features.py, run_task_a.py,
run_task_b.py, feature_importance.py, compute_diagnostics.py, generate_report_pdf.py, run_all.py --
they all import their paths/geometry from here). Unset = exactly the original full-resolution run
(embeddings/, results/, 1920x1184 crop, 120x74 patch grid) -- untouched by any variant run.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent

VARIANT = os.environ.get("DINO_VARIANT", "").strip()
DOWNSCALE = float(os.environ.get("DINO_DOWNSCALE", "1.0"))
_SUFFIX = f"_{VARIANT}" if VARIANT else ""

EMB_DIR = ROOT / f"embeddings{_SUFFIX}"
RESULTS_DIR = ROOT / f"results{_SUFFIX}"
PLOTS_DIR = RESULTS_DIR / "plots"
METRICS_CSV = RESULTS_DIR / "metrics.csv"

MANIFEST = EMB_DIR / "manifest.parquet"
POOLED_NPY = EMB_DIR / "pooled.npy"
PATCH_NPY = EMB_DIR / "patch.npy"
SPECIAL_NPY = EMB_DIR / "special.npy"
EMB_META = EMB_DIR / "emb_meta.json"

# --- image geometry (single source of truth, derived from DOWNSCALE) ---
RAW_SIZE = (1920, 1275)                 # original TIFF (W, H)
BASE_CROP_W, BASE_CROP_H = 1920, 1184   # after removing the ~91px instrument databar, full-res
CROP_W = int(round(BASE_CROP_W * DOWNSCALE / 16) * 16)
CROP_H = int(round(BASE_CROP_H * DOWNSCALE / 16) * 16)
assert CROP_W > 0 and CROP_H > 0, f"downscale {DOWNSCALE} collapses crop to zero"
GRID_W, GRID_H = CROP_W // 16, CROP_H // 16
NUM_PATCHES = GRID_H * GRID_W

IDENTIFIER_COLS = ["lsv_overpotential_mV", "Experiment_data", "Sample_id"]
NON_FEATURE_META_COLS = [
    "image_id", "rel_path", "session_folder", "experiment_date", "sample_id", "sample_key",
    "magnification", "detector", "kv", "image_number", "emb_index", "grid_h", "grid_w",
]
PRIMARY_BANDS = [2000, 10000, 40000]  # near-universal magnifications (>=296/298 samples)

SEED = 42
N_FOLDS = 5


def ensure_dirs():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_manifest() -> pd.DataFrame:
    return pd.read_parquet(MANIFEST).sort_values("emb_index").reset_index(drop=True)


def synthesis_feature_cols(df: pd.DataFrame) -> list:
    excl = set(IDENTIFIER_COLS) | set(NON_FEATURE_META_COLS)
    cand = [c for c in df.columns if c not in excl]
    keep = [c for c in cand if df[c].nunique(dropna=True) > 1]
    dropped = [c for c in cand if c not in keep]
    if dropped:
        print(f"[exp_common] dropping {len(dropped)} zero-variance synth cols: {dropped}")
    return keep


def sample_table(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per sample_key: 51 synthesis features + OER target."""
    if df is None:
        df = load_manifest()
    feat_cols = synthesis_feature_cols(df)
    g = df.groupby("sample_key", sort=True)
    nun = g["lsv_overpotential_mV"].nunique()
    assert (nun <= 1).all(), "OER target is not unique per sample -- investigate join."
    samp = g[feat_cols + ["lsv_overpotential_mV"]].first().reset_index()
    return samp, feat_cols


def load_pooled() -> np.ndarray:
    return np.load(POOLED_NPY, mmap_mode="r")


def load_patch() -> np.ndarray:
    return np.load(PATCH_NPY, mmap_mode="r")


def load_special() -> np.ndarray:
    return np.load(SPECIAL_NPY, mmap_mode="r")


# ---------------- folds ----------------

def make_fold_ids(n: int, n_folds: int = N_FOLDS, seed: int = SEED) -> np.ndarray:
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_of = np.full(n, -1, dtype=int)
    for fold_id, (_, test_idx) in enumerate(kf.split(np.arange(n))):
        fold_of[test_idx] = fold_id
    return fold_of


def sample_fold_map(sample_keys_unique, n_folds: int = N_FOLDS, seed: int = SEED) -> dict:
    """The ONE canonical sample->fold assignment used by every experiment (Task A and B alike)."""
    keys = sorted(sample_keys_unique)
    folds = make_fold_ids(len(keys), n_folds, seed)
    return dict(zip(keys, folds))


def image_fold_array(df: pd.DataFrame, fold_map: dict) -> np.ndarray:
    arr = df["sample_key"].map(fold_map).to_numpy()
    assert not pd.isna(arr).any(), "some images have no fold assignment"
    return arr.astype(int)


# ---------------- generic scalar-target CV ----------------

def _impute_nan_with_train_mean(Xtr, Xte):
    col_means = np.nanmean(Xtr, axis=0)
    col_means = np.nan_to_num(col_means, nan=0.0)
    Xtr = np.where(np.isnan(Xtr), col_means[None, :], Xtr)
    Xte = np.where(np.isnan(Xte), col_means[None, :], Xte)
    return Xtr, Xte


def cv_predict_scalar(X, y, folds, model_factory, pca_components=None, pca_cols=None, seed: int = SEED):
    """5-fold CV for a scalar target. X:(N,F) y:(N,) folds:(N,) int in [0,n_folds).

    pca_components: if set (and pca_cols is None), PCA-reduce the WHOLE feature matrix to this
      many components (only when F > 100; skipped otherwise) before fitting.
    pca_cols: optional column indices to PCA-reduce; remaining columns pass through raw (used to
      keep interpretable low-dim synthesis columns un-reduced while compressing embedding blocks).
    All scalers/PCA are fit on the training fold only.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    n_folds = int(folds.max()) + 1
    oof_pred = np.full(len(y), np.nan)
    fold_metrics = []

    for f in range(n_folds):
        test_mask = folds == f
        train_mask = ~test_mask
        Xtr, Xte = X[train_mask].copy(), X[test_mask].copy()
        ytr, yte = y[train_mask], y[test_mask]
        Xtr, Xte = _impute_nan_with_train_mean(Xtr, Xte)

        if pca_cols is not None:
            other = [c for c in range(X.shape[1]) if c not in set(pca_cols)]
            s_pca = StandardScaler().fit(Xtr[:, pca_cols])
            tr_pca, te_pca = s_pca.transform(Xtr[:, pca_cols]), s_pca.transform(Xte[:, pca_cols])
            n_comp = max(1, min(pca_components or 32, tr_pca.shape[0] - 1, tr_pca.shape[1]))
            pca = PCA(n_components=n_comp, random_state=seed).fit(tr_pca)
            Xtr = np.hstack([Xtr[:, other], pca.transform(tr_pca)])
            Xte = np.hstack([Xte[:, other], pca.transform(te_pca)])
        elif pca_components and X.shape[1] > 100:
            s0 = StandardScaler().fit(Xtr)
            tr0, te0 = s0.transform(Xtr), s0.transform(Xte)
            n_comp = max(1, min(pca_components, tr0.shape[0] - 1, tr0.shape[1]))
            pca = PCA(n_components=n_comp, random_state=seed).fit(tr0)
            Xtr, Xte = pca.transform(tr0), pca.transform(te0)

        scaler = StandardScaler().fit(Xtr)
        Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)

        model = model_factory()
        model.fit(Xtr_s, ytr)
        pred = model.predict(Xte_s)
        oof_pred[test_mask] = pred
        fold_metrics.append(regression_metrics(yte, pred))

    return fold_metrics, oof_pred


# ---------------- metrics ----------------

def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    if len(y_true) >= 2 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        pear = float(pearsonr(y_true, y_pred)[0])
        spear = float(spearmanr(y_true, y_pred)[0])
    else:
        pear = spear = float("nan")
    return {"r2": float(r2), "rmse": rmse, "mae": mae, "pearson": pear, "spearman": spear, "n": len(y_true)}


def multioutput_metrics(y_true, y_pred) -> dict:
    """y_true, y_pred: (N, D). Per-dim R2 summary + mean cosine similarity + explained variance."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    ss_tot = np.sum((y_true - y_true.mean(axis=0, keepdims=True)) ** 2, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        per_dim_r2 = 1 - ss_res / ss_tot
    per_dim_r2 = per_dim_r2[np.isfinite(per_dim_r2)]
    overall_r2 = float(1 - ss_res.sum() / ss_tot.sum()) if ss_tot.sum() > 0 else float("nan")
    num = np.sum(y_true * y_pred, axis=1)
    den = np.linalg.norm(y_true, axis=1) * np.linalg.norm(y_pred, axis=1)
    den = np.where(den == 0, np.nan, den)
    cos = num / den
    explained_var = 1 - np.var(y_true - y_pred) / np.var(y_true)
    return {
        "mean_r2": float(np.mean(per_dim_r2)) if len(per_dim_r2) else float("nan"),
        "median_r2": float(np.median(per_dim_r2)) if len(per_dim_r2) else float("nan"),
        "overall_r2": overall_r2,
        "frac_dims_r2_gt0": float(np.mean(per_dim_r2 > 0)) if len(per_dim_r2) else float("nan"),
        "mean_cosine_sim": float(np.nanmean(cos)),
        "explained_variance": float(explained_var),
        "n": int(y_true.shape[0]),
        "d": int(y_true.shape[1]),
    }


# ---------------- reporting ----------------

def log_result(experiment: str, variant: str, model: str, fold_metrics: list, extra: dict | None = None) -> dict:
    """Aggregate per-fold metric dicts (mean+/-std) and upsert into results/metrics.csv + write JSON."""
    ensure_dirs()
    keys = [k for k in fold_metrics[0].keys() if k != "n"]
    agg = {"experiment": experiment, "variant": variant, "model": model, "n_folds": len(fold_metrics)}
    for k in keys:
        vals = np.array([m[k] for m in fold_metrics], dtype=float)
        agg[f"{k}_mean"] = float(np.nanmean(vals))
        agg[f"{k}_std"] = float(np.nanstd(vals))
    if extra:
        agg.update(extra)

    row = pd.DataFrame([agg])
    if METRICS_CSV.exists():
        old = pd.read_csv(METRICS_CSV)
        key_cols = ["experiment", "variant", "model"]
        mask = np.ones(len(old), dtype=bool)
        for c in key_cols:
            mask &= (old[c].astype(str) == str(agg[c]))
        old = old[~mask]
        combined = pd.concat([old, row], ignore_index=True)
    else:
        combined = row
    combined.to_csv(METRICS_CSV, index=False)

    safe_name = f"{experiment}__{variant}__{model}".replace(" ", "_").replace("/", "-")
    json_path = RESULTS_DIR / f"{safe_name}.json"
    json_path.write_text(json.dumps({"summary": agg, "folds": fold_metrics}, indent=2))
    return agg
