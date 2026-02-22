#!/usr/bin/env python3
"""
ensemble_mk_trends.py
Multimodel ensemble aggregation of Mann-Kendall trend results.

Reads:  mk_trends_{model}_{ssp}.nc  (one per model, from compute_mk_trends.py)
Output: mk_ensemble_{ssp}_isimip3b.nc

  Dimensions : (lat, lon)
  Variables  : {window}_{metric}_{ens_mean|ens_std|agreement|robust|sig_fraction}

Usage:
  python ensemble_mk_trends.py --ssp ssp126 \
      --mk-dir ~/drought_catalog --out-dir ~/drought_catalog

  # All scenarios:
  for ssp in ssp126 ssp370 ssp585; do
      python ensemble_mk_trends.py --ssp $ssp \
          --mk-dir ~/drought_catalog --out-dir ~/drought_catalog
  done
"""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr

# =============================================================================
# Config
# =============================================================================

MODELS = [
    "gfdl-esm4",
    "ipsl-cm6a-lr",
    "mpi-esm1-2-hr",
    "mri-esm2-0",
    "ukesm1-0-ll",
]

WINDOWS  = ["ref", "near", "far"]
METRICS  = ["drought_count", "ann_min_spei", "ann_sum_def"]

ROBUST_THRESHOLD = 0.8   # fraction of models that must agree on sign


# =============================================================================
# IO
# =============================================================================

def open_nc(path):
    for engine in ("h5netcdf", "scipy", "netcdf4"):
        try:
            return xr.open_dataset(path, engine=engine)
        except Exception:
            continue
    raise RuntimeError(f"Cannot open: {path}")


# =============================================================================
# Ensemble aggregation
# =============================================================================

def aggregate(slopes, pvals=None, alpha=0.05):
    """
    slopes : (n_models, n_lat, n_lon)
    pvals  : (n_models, n_lat, n_lon) or None

    Returns mean, std, agreement, robust, sig_fraction — all (n_lat, n_lon).
    """
    ens_mean  = np.nanmean(slopes, axis=0)
    ens_std   = np.nanstd(slopes,  axis=0)

    mean_sign = np.sign(ens_mean)
    n_agree   = np.nansum(
        np.sign(slopes) == mean_sign[np.newaxis], axis=0
    ).astype(float)
    n_valid   = np.sum(np.isfinite(slopes), axis=0).astype(float)
    agreement = np.where(n_valid > 0, n_agree / n_valid, np.nan)
    robust    = np.where(
        (agreement >= ROBUST_THRESHOLD) & np.isfinite(ens_mean),
        np.float32(1.0), np.float32(0.0)
    )

    if pvals is not None:
        sig_fraction = np.nanmean(
            (pvals < alpha).astype(float), axis=0
        )
    else:
        sig_fraction = np.full_like(ens_mean, np.nan)

    return (ens_mean.astype(np.float32), ens_std.astype(np.float32),
            agreement.astype(np.float32), robust.astype(np.float32),
            sig_fraction.astype(np.float32))


# =============================================================================
# Main
# =============================================================================

def run_ensemble(ssp, mk_dir, out_dir):
    mk_dir  = Path(mk_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"MK ENSEMBLE AGGREGATION: {ssp.upper()}")
    print(f"{'='*60}")

    # ── Load per-model files ──────────────────────────────────────────────────
    model_datasets = []
    models_loaded  = []
    for model in MODELS:
        path = mk_dir / f"mk_trends_{model}_{ssp}.nc"
        if not path.exists():
            print(f"  ⚠ Missing: {path.name}")
            continue
        try:
            ds = open_nc(path)
            model_datasets.append(ds)
            models_loaded.append(model)
            print(f"  ✓ {model}  (grid {ds.sizes['lat']}×{ds.sizes['lon']})")
        except Exception as e:
            print(f"  ✗ {model} — {e}")

    if len(model_datasets) < 2:
        raise RuntimeError(
            f"Need ≥2 models, only {len(model_datasets)} loaded.")

    n_models = len(model_datasets)
    lats     = model_datasets[0]["lat"].values
    lons     = model_datasets[0]["lon"].values
    print(f"\n  Models loaded : {n_models}  ({', '.join(models_loaded)})")
    print(f"  Grid          : {len(lats)}×{len(lons)}")

    # ── Aggregate all window × metric combinations ────────────────────────────
    data_vars = {}

    for wk in WINDOWS:
        for metric in METRICS:
            slope_var = f"{wk}_{metric}_slope"
            p_var     = f"{wk}_{metric}_p"

            slopes = np.stack([
                ds[slope_var].values.astype(float)
                for ds in model_datasets
            ], axis=0)   # (n_models, n_lat, n_lon)

            pvals = np.stack([
                ds[p_var].values.astype(float)
                for ds in model_datasets
                if p_var in ds.data_vars
            ], axis=0) if p_var in model_datasets[0].data_vars else None

            mean, std, agr, rob, sig = aggregate(slopes, pvals)

            base = f"{wk}_{metric}"
            data_vars[f"{base}_ens_mean"]     = (["lat", "lon"], mean,
                {"long_name": f"Ensemble mean slope — {base}"})
            data_vars[f"{base}_ens_std"]      = (["lat", "lon"], std,
                {"long_name": f"Ensemble std slope — {base}"})
            data_vars[f"{base}_agreement"]    = (["lat", "lon"], agr,
                {"long_name": f"Sign agreement fraction — {base}"})
            data_vars[f"{base}_robust"]       = (["lat", "lon"], rob,
                {"long_name": f"Robust signal mask — {base}",
                 "comment":   f"1 where agreement >= {ROBUST_THRESHOLD}"})
            data_vars[f"{base}_sig_fraction"] = (["lat", "lon"], sig,
                {"long_name": f"Fraction of models p<alpha — {base}"})

    # ── Build & save dataset ──────────────────────────────────────────────────
    ds_out = xr.Dataset(
        data_vars,
        coords={"lat": lats, "lon": lons},
        attrs={
            "title":            "Multimodel Ensemble MK Trends — SPEI-02",
            "scenario":         ssp,
            "n_models":         n_models,
            "models":           ", ".join(models_loaded),
            "mk_method":        "Yue-Wang TFPW",
            "protocol":         "ISIMIP3b",
            "robust_threshold": ROBUST_THRESHOLD,
            "agreement_note":   (
                "agreement = fraction of models with same sign as ensemble mean. "
                f"robust = 1 where agreement >= {ROBUST_THRESHOLD}"
            ),
            "created_utc":      datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "conventions":      "CF-1.8",
        },
    )

    out_path = out_dir / f"mk_ensemble_{ssp}_isimip3b.nc"
    enc = {v: {"zlib": True, "complevel": 4, "dtype": "float32"}
           for v in ds_out.data_vars}
    ds_out.to_netcdf(out_path, encoding=enc)
    print(f"\n  ✓ Saved: {out_path.name}  "
          f"({out_path.stat().st_size / 1e6:.1f} MB)")
    print(f"  Variables: {len(list(ds_out.data_vars))}")

    # ── Quick summary ─────────────────────────────────────────────────────────
    print(f"\n  Robust signal summary (% cells, ≥{ROBUST_THRESHOLD*100:.0f}% agreement):")
    print(f"  {'Variable':<30} {'ref':>7} {'near':>7} {'far':>7}")
    print(f"  {'-'*53}")
    for metric in METRICS:
        row = f"  {metric:<30}"
        for wk in WINDOWS:
            rob = ds_out[f"{wk}_{metric}_robust"].values
            fin = np.isfinite(ds_out[f"{wk}_{metric}_ens_mean"].values)
            pct = rob[fin].mean() * 100 if fin.sum() > 0 else 0.0
            row += f" {pct:>6.1f}%"
        print(row)

    for ds in model_datasets:
        ds.close()

    return ds_out


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Multimodel ensemble aggregation of MK trend results."
    )
    p.add_argument("--ssp",      required=True,
                   help="Scenario: ssp126 | ssp370 | ssp585")
    p.add_argument("--mk-dir",   required=True,
                   help="Directory with mk_trends_{model}_{ssp}.nc files")
    p.add_argument("--out-dir",  required=True,
                   help="Output directory for ensemble NetCDF")
    p.add_argument("--robust-threshold", default=ROBUST_THRESHOLD, type=float,
                   help=f"Min agreement fraction for robust signal (default {ROBUST_THRESHOLD})")
    return p.parse_args()


def main():
    args = parse_args()
    global ROBUST_THRESHOLD
    ROBUST_THRESHOLD = args.robust_threshold
    run_ensemble(args.ssp, args.mk_dir, args.out_dir)


if __name__ == "__main__":
    main()