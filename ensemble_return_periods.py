#!/usr/bin/env python3
"""
ensemble_return_periods.py
Multimodel ensemble aggregation of GPD/POT return period results.

Reads:  return_periods_{model}_{ssp}.nc  (one per model, from compute_return_periods.py)
Output: return_periods_ensemble_{ssp}.nc

  Dimensions: cell, return_period
  Variables (per climate metric):
    lat, lon
    {metric}_ens_mean     ensemble mean across models
    {metric}_ens_std      model spread (std)
    {metric}_agreement    fraction of models with same sign as ensemble mean
    {metric}_robust       1 where agreement >= robust_threshold

  Aggregated metrics (per variable=intensity/severity):
    delta_sev_abs_{var}_{rp}
    delta_sev_rel_{var}_{rp}
    rp_ratio_{var}_{rp}
    delta_lambda_{var}_abs
    delta_lambda_{var}_rel

Usage:
  python ensemble_return_periods.py --ssp ssp126 \
      --rp-dir ~/drought_catalog \
      --out-dir ~/drought_catalog

  # All scenarios:
  for ssp in ssp126 ssp370 ssp585; do
      python ensemble_return_periods.py --ssp $ssp \
          --rp-dir ~/drought_catalog --out-dir ~/drought_catalog
  done
"""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
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

# Climate metrics to aggregate — scalar per cell
SCALAR_METRICS = [
    "delta_lambda_{var}_abs",
    "delta_lambda_{var}_rel",
]

# Metrics that also carry a return period dimension
RP_METRICS = [
    "delta_sev_abs_{var}_{rp}",
    "delta_sev_rel_{var}_{rp}",
    "rp_far_equiv_{var}_{rp}",
    "rp_ratio_{var}_{rp}",
]

VARS = ["intensity", "severity"]

ROBUST_THRESHOLD = 0.8   # fraction of models that must agree on sign


# =============================================================================
# IO helpers
# =============================================================================

def open_nc(path):
    for engine in ("h5netcdf", "scipy", "netcdf4"):
        try:
            return xr.open_dataset(path, engine=engine)
        except Exception:
            continue
    raise RuntimeError(f"Cannot open: {path}")


# =============================================================================
# Cell alignment
# =============================================================================

def ds_to_df_index(ds):
    """Return (lat_r, lon_r) rounded coordinate pairs as a DataFrame index."""
    lats = np.round(ds["lat"].values.astype(float), 4)
    lons = np.round(ds["lon"].values.astype(float), 4)
    return pd.MultiIndex.from_arrays([lats, lons], names=["lat_r", "lon_r"])


def align_models(model_datasets):
    """
    Find the intersection of (lat, lon) cells present in ALL loaded models.
    Returns a sorted list of (lat_r, lon_r) tuples.
    """
    indices = [set(zip(
        np.round(ds["lat"].values.astype(float), 4),
        np.round(ds["lon"].values.astype(float), 4)
    )) for ds in model_datasets]

    common = indices[0]
    for idx in indices[1:]:
        common &= idx

    return sorted(common)


def extract_cell_values(ds, common_cells, variable):
    """
    Extract values for `variable` at the common cell (lat, lon) positions.
    Returns 1-D array aligned to common_cells order.
    """
    lats = np.round(ds["lat"].values.astype(float), 4)
    lons = np.round(ds["lon"].values.astype(float), 4)
    coord_to_idx = {(la, lo): i for i, (la, lo) in enumerate(zip(lats, lons))}

    out = np.full(len(common_cells), np.nan, dtype=np.float32)
    for j, cell in enumerate(common_cells):
        idx = coord_to_idx.get(cell)
        if idx is not None and variable in ds.data_vars:
            out[j] = float(ds[variable].values[idx])
    return out


def extract_rp_values(ds, common_cells, variable):
    """
    Extract (cell × return_period) matrix for `variable`.
    Returns 2-D array (n_cells, n_rp).
    """
    lats = np.round(ds["lat"].values.astype(float), 4)
    lons = np.round(ds["lon"].values.astype(float), 4)
    coord_to_idx = {(la, lo): i for i, (la, lo) in enumerate(zip(lats, lons))}

    n_rp = ds.sizes.get("return_period", 0)
    out  = np.full((len(common_cells), n_rp), np.nan, dtype=np.float32)

    for j, cell in enumerate(common_cells):
        idx = coord_to_idx.get(cell)
        if idx is not None and variable in ds.data_vars:
            out[j] = ds[variable].values[idx]
    return out


# =============================================================================
# Ensemble aggregation
# =============================================================================

def aggregate_1d(stacked):
    """
    stacked : (n_models, n_cells)  — each row is one model's values.
    Returns mean, std, agreement, robust arrays each shape (n_cells,).
    """
    ens_mean = np.nanmean(stacked, axis=0)
    ens_std  = np.nanstd(stacked,  axis=0)

    mean_sign = np.sign(ens_mean)
    n_agree   = np.nansum(np.sign(stacked) == mean_sign[np.newaxis, :], axis=0).astype(float)
    n_valid   = np.sum(np.isfinite(stacked), axis=0).astype(float)
    agreement = np.where(n_valid > 0, n_agree / n_valid, np.nan)
    robust    = np.where(
        (agreement >= ROBUST_THRESHOLD) & np.isfinite(ens_mean),
        np.float32(1.0), np.float32(0.0)
    )
    return ens_mean.astype(np.float32), ens_std.astype(np.float32), \
           agreement.astype(np.float32), robust.astype(np.float32)


def aggregate_2d(stacked_list):
    """
    stacked_list : list of (n_cells, n_rp) arrays, one per model.
    Returns mean, std, agreement, robust each shape (n_cells, n_rp).
    """
    cube     = np.stack(stacked_list, axis=0)   # (n_models, n_cells, n_rp)
    ens_mean = np.nanmean(cube, axis=0)
    ens_std  = np.nanstd(cube,  axis=0)

    mean_sign = np.sign(ens_mean)
    n_agree   = np.nansum(np.sign(cube) == mean_sign[np.newaxis], axis=0).astype(float)
    n_valid   = np.sum(np.isfinite(cube), axis=0).astype(float)
    agreement = np.where(n_valid > 0, n_agree / n_valid, np.nan)
    robust    = np.where(
        (agreement >= ROBUST_THRESHOLD) & np.isfinite(ens_mean),
        np.float32(1.0), np.float32(0.0)
    )
    return (ens_mean.astype(np.float32), ens_std.astype(np.float32),
            agreement.astype(np.float32), robust.astype(np.float32))


# =============================================================================
# Main aggregation
# =============================================================================

def run_ensemble(ssp, rp_dir, out_dir):
    rp_dir  = Path(rp_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"ENSEMBLE RETURN PERIODS: {ssp.upper()}")
    print(f"{'='*60}")

    # ── Load per-model files ──────────────────────────────────────────────────
    model_datasets = []
    models_loaded  = []
    for model in MODELS:
        path = rp_dir / f"return_periods_{model}_{ssp}.nc"
        if not path.exists():
            print(f"  ⚠ Missing: {path.name}")
            continue
        try:
            ds = open_nc(path)
            model_datasets.append(ds)
            models_loaded.append(model)
            print(f"  ✓ {model}  ({ds.sizes['cell']:,} cells)")
        except Exception as e:
            print(f"  ✗ {model} — {e}")

    if len(model_datasets) < 2:
        raise RuntimeError(f"Need ≥2 models, only {len(model_datasets)} loaded.")

    # Grab return period values from first dataset
    return_periods = list(model_datasets[0]["return_period"].values.astype(int))
    n_rp           = len(return_periods)
    n_models       = len(model_datasets)
    print(f"\n  Models loaded : {n_models}  ({', '.join(models_loaded)})")
    print(f"  Return periods: {return_periods}")

    # ── Find common cells ─────────────────────────────────────────────────────
    common_cells = align_models(model_datasets)
    n_cells      = len(common_cells)
    lats_out     = np.array([c[0] for c in common_cells], dtype=np.float32)
    lons_out     = np.array([c[1] for c in common_cells], dtype=np.float32)
    print(f"  Common cells  : {n_cells:,}")

    # ── Aggregate ─────────────────────────────────────────────────────────────
    data_vars = {
        "lat": (["cell"], lats_out, {"units": "degrees_north"}),
        "lon": (["cell"], lons_out, {"units": "degrees_east"}),
    }

    for var in VARS:

        # Scalar metrics (lambda change etc.)
        for tmpl in SCALAR_METRICS:
            vname = tmpl.format(var=var)
            stacked = np.stack(
                [extract_cell_values(ds, common_cells, vname)
                 for ds in model_datasets], axis=0
            )
            mean, std, agr, rob = aggregate_1d(stacked)
            _add_ensemble_vars(data_vars, vname, mean, std, agr, rob)

        # RP-dimensioned metrics
        for tmpl in RP_METRICS:
            for rp in return_periods:
                vname = tmpl.format(var=var, rp=rp)
                stacked = np.stack(
                    [extract_cell_values(ds, common_cells, vname)
                     for ds in model_datasets], axis=0
                )
                mean, std, agr, rob = aggregate_1d(stacked)
                _add_ensemble_vars(data_vars, vname, mean, std, agr, rob)

        # GPD quantile matrix (cell × return_period)
        for sl in ("hist", "far"):
            vname = f"rp_quantile_{var}_{sl}"
            mats  = [extract_rp_values(ds, common_cells, vname)
                     for ds in model_datasets]
            mean, std, agr, rob = aggregate_2d(mats)
            _add_ensemble_vars_2d(data_vars, vname, mean, std, agr, rob)

    # ── Build dataset ─────────────────────────────────────────────────────────
    ds_out = xr.Dataset(
        data_vars,
        coords={
            "cell":          np.arange(n_cells, dtype=np.int32),
            "return_period": np.array(return_periods, dtype=np.int32),
        },
        attrs={
            "title":            "Multimodel Ensemble Return Periods (GPD/POT)",
            "scenario":         ssp,
            "n_models":         n_models,
            "models":           ", ".join(models_loaded),
            "robust_threshold": ROBUST_THRESHOLD,
            "agreement_note":   (
                "agreement = fraction of models with same sign as ensemble mean. "
                f"robust = 1 where agreement >= {ROBUST_THRESHOLD}"
            ),
            "created_utc":      datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "conventions":      "CF-1.8",
        },
    )

    out_path = out_dir / f"return_periods_ensemble_{ssp}.nc"
    enc = {v: {"zlib": True, "complevel": 4}
           for v in ds_out.data_vars
           if ds_out[v].dtype in (np.float32, np.int32, np.int8)}
    ds_out.to_netcdf(out_path, encoding=enc)
    print(f"\n  ✓ Saved: {out_path.name}  "
          f"({out_path.stat().st_size / 1e6:.1f} MB)")
    print(f"  Variables: {len(list(ds_out.data_vars))}")

    # ── Quick summary ─────────────────────────────────────────────────────────
    print(f"\n  Robust signal summary (% cells, ≥{ROBUST_THRESHOLD*100:.0f}% agreement):")
    print(f"  {'Variable':<35} {'Robust %':>9}")
    print(f"  {'-'*46}")
    for var in VARS:
        for rp in [10, 50, 100]:
            vname = f"delta_sev_abs_{var}_{rp}_robust"
            if vname in ds_out:
                pct = float(np.nanmean(ds_out[vname].values)) * 100
                print(f"  delta_sev_abs_{var}_RP{rp:<4}         {pct:>8.1f}%")
        lname = f"delta_lambda_{var}_abs_robust"
        if lname in ds_out:
            pct = float(np.nanmean(ds_out[lname].values)) * 100
            print(f"  delta_lambda_{var}_abs              {pct:>8.1f}%")

    for ds in model_datasets:
        ds.close()

    return ds_out


# =============================================================================
# Helpers
# =============================================================================

def _add_ensemble_vars(data_vars, base, mean, std, agr, rob):
    data_vars[f"{base}_ens_mean"]  = (["cell"], mean,
        {"long_name": f"Ensemble mean — {base}"})
    data_vars[f"{base}_ens_std"]   = (["cell"], std,
        {"long_name": f"Ensemble std — {base}"})
    data_vars[f"{base}_agreement"] = (["cell"], agr,
        {"long_name": f"Sign agreement fraction — {base}"})
    data_vars[f"{base}_robust"]    = (["cell"], rob,
        {"long_name": f"Robust signal mask — {base}",
         "comment": f"1 where agreement >= {ROBUST_THRESHOLD}"})


def _add_ensemble_vars_2d(data_vars, base, mean, std, agr, rob):
    data_vars[f"{base}_ens_mean"]  = (["cell", "return_period"], mean,
        {"long_name": f"Ensemble mean — {base}"})
    data_vars[f"{base}_ens_std"]   = (["cell", "return_period"], std,
        {"long_name": f"Ensemble std — {base}"})
    data_vars[f"{base}_agreement"] = (["cell", "return_period"], agr,
        {"long_name": f"Sign agreement fraction — {base}"})
    data_vars[f"{base}_robust"]    = (["cell", "return_period"], rob,
        {"long_name": f"Robust signal mask — {base}"})


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Multimodel ensemble aggregation of GPD return period results."
    )
    p.add_argument("--ssp",      required=True,
                   help="Scenario: ssp126 | ssp370 | ssp585")
    p.add_argument("--rp-dir",   required=True,
                   help="Directory with return_periods_{model}_{ssp}.nc files")
    p.add_argument("--out-dir",  required=True,
                   help="Output directory for ensemble NetCDF")
    p.add_argument("--robust-threshold", default=ROBUST_THRESHOLD, type=float,
                   help=f"Min agreement fraction for robust signal (default {ROBUST_THRESHOLD})")
    return p.parse_args()


def main():
    args = parse_args()
    global ROBUST_THRESHOLD
    ROBUST_THRESHOLD = args.robust_threshold
    run_ensemble(args.ssp, args.rp_dir, args.out_dir)


if __name__ == "__main__":
    main()