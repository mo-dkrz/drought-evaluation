#!/usr/bin/env python3
"""
compute_mk_trends.py
Mann-Kendall trend analysis on SPEI-02 over ISIMIP3b windows (all grid cells).

WINDOWS (ISIMIP3b protocol):
  ref  : 1979-2014  (36 yr, satellite-era anchor)
  near : 2025-2055  (31 yr, actionable insurance horizon)
  far  : 2055-2085  (31 yr, long-term equilibrium)

METRICS (computed per year, per cell):
  drought_count  — N months SPEI < threshold per year
  ann_min_spei   — annual minimum SPEI (most extreme month)
  ann_sum_def    — annual sum |SPEI| below threshold (severity)

MK TEST: Yue-Wang modification (TFPW) via pymannkendall.

Output: mk_trends_{model}_{ssp}.nc
  Dimensions : (lat, lon)
  Variables  : {window}_{metric}_{slope|p|tau|significant|n_years}

Usage:
  python compute_mk_trends.py --model gfdl-esm4 --ssp ssp126 \\
      --input-dir ~/spei_r_outputs --out-dir ~/drought_catalog
"""

import argparse
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

try:
    import pymannkendall as mk_lib
except ImportError:
    raise ImportError("pip install pymannkendall")


# =============================================================================
# ISIMIP3b window definitions
# =============================================================================

WINDOWS = {
    "ref":  {"start": "1979-01-01", "end": "2014-12-31", "n_years": 36,
             "label": "Reference (1979–2014)"},
    "near": {"start": "2025-01-01", "end": "2055-12-31", "n_years": 31,
             "label": "Near-Future (2025–2055)"},
    "far":  {"start": "2055-01-01", "end": "2085-12-31", "n_years": 31,
             "label": "Far-Future (2055–2085)"},
}

METRICS   = ["drought_count", "ann_min_spei", "ann_sum_def"]
MIN_YEARS = 10     # minimum annual values for a valid MK test


# =============================================================================
# Core functions
# =============================================================================

def annual_series(da_window: xr.DataArray, threshold: float) -> pd.DataFrame:
    """Monthly SPEI slice → annual drought metrics DataFrame."""
    df = pd.DataFrame({
        "spei": da_window.values.astype(float),
        "time": pd.to_datetime(da_window["time"].values),
    })
    df["year"]    = df["time"].dt.year
    df["drought"] = df["spei"] < threshold
    return df.groupby("year").agg(
        drought_count=("drought", "sum"),
        ann_min_spei =("spei",    "min"),
        ann_sum_def  =("spei",    lambda x: x[x < threshold].abs().sum()),
    )


def run_mk_cell(spei_1d: np.ndarray, times: np.ndarray,
                window_key: str, threshold: float) -> dict:
    """
    MK test for one cell × one window.
    spei_1d : 1-D float array (full time series)
    times   : corresponding datetime64 array
    Returns dict with slope, p, tau, n per metric.
    """
    w       = WINDOWS[window_key]
    t_idx   = pd.to_datetime(times)
    mask    = (t_idx >= w["start"]) & (t_idx <= w["end"])
    spei_sl = spei_1d[mask]
    time_sl = times[mask]

    if spei_sl.size == 0 or np.all(np.isnan(spei_sl)):
        return {m: {"slope": np.nan, "p": np.nan, "tau": np.nan, "n": 0}
                for m in METRICS}

    da_sl  = xr.DataArray(spei_sl, coords={"time": time_sl}, dims=["time"])
    ann    = annual_series(da_sl, threshold)
    result = {}

    for metric in METRICS:
        series = ann[metric].values
        series = series[np.isfinite(series)]

        if len(series) < MIN_YEARS:
            result[metric] = {"slope": np.nan, "p": np.nan,
                              "tau": np.nan, "n": len(series)}
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = mk_lib.yue_wang_modification_test(series)
            result[metric] = {
                "slope": res.slope, "p": res.p,
                "tau": res.Tau,     "n": len(series),
            }
        except Exception:
            result[metric] = {"slope": np.nan, "p": np.nan,
                              "tau": np.nan,   "n": len(series)}
    return result


# =============================================================================
# Full-grid computation
# =============================================================================

def run_full_grid(da: xr.DataArray, threshold: float,
                  alpha: float) -> tuple[dict, np.ndarray, np.ndarray]:
    """
    Loop over all land cells × all windows × all metrics.
    Returns (arrays dict, lats, lons).
    """
    lats  = da["lat"].values
    lons  = da["lon"].values
    n_lat = len(lats)
    n_lon = len(lons)
    times = da["time"].values   # datetime64 array — load once

    stats   = ["slope", "p", "tau", "significant", "n_years"]
    windows = list(WINDOWS.keys())

    arrays = {}
    for wk in windows:
        for m in METRICS:
            for s in stats:
                arrays[f"{wk}_{m}_{s}"] = np.full(
                    (n_lat, n_lon), np.nan, dtype=np.float32)

    print(f"  Grid   : {n_lat}×{n_lon} = {n_lat*n_lon:,} cells")
    print(f"  Windows: {windows}")
    print(f"  Metrics: {METRICS}")

    # Load data once (avoids repeated xarray indexing overhead)
    da_np = da.values.astype(np.float32)   # (time, lat, lon)
    n_skipped = 0

    for i_lat in tqdm(range(n_lat), desc="  Rows"):
        for i_lon in range(n_lon):
            spei_1d = da_np[:, i_lat, i_lon]

            # Skip ocean / all-NaN cells
            # (don't use index 0 — SPEI-02 initialization leaves first
            #  1-2 timesteps NaN even over land)
            if np.all(np.isnan(spei_1d)):
                n_skipped += 1
                continue

            for wk in windows:
                res = run_mk_cell(spei_1d, times, wk, threshold)
                for metric, r in res.items():
                    base = f"{wk}_{metric}"
                    arrays[f"{base}_slope"][i_lat, i_lon]       = r["slope"]
                    arrays[f"{base}_p"][i_lat, i_lon]           = r["p"]
                    arrays[f"{base}_tau"][i_lat, i_lon]         = r["tau"]
                    arrays[f"{base}_n_years"][i_lat, i_lon]     = r["n"]
                    arrays[f"{base}_significant"][i_lat, i_lon] = (
                        1.0 if (np.isfinite(r["p"]) and r["p"] < alpha)
                        else 0.0
                    )

    print(f"  ✓ {n_skipped:,} ocean/NaN cells skipped")
    return arrays, lats, lons


# =============================================================================
# IO
# =============================================================================

def load_spei(spei_path: Path, spei_var: str) -> xr.DataArray:
    if not spei_path.exists():
        raise FileNotFoundError(f"SPEI file not found: {spei_path}")
    for engine in ("h5netcdf", "scipy", "netcdf4"):
        try:
            ds = xr.open_dataset(spei_path, engine=engine,
                                 chunks={"time": -1, "lat": 25, "lon": 25})
            if spei_var not in ds.data_vars:
                raise KeyError(
                    f"'{spei_var}' not in file. Found: {list(ds.data_vars)}")
            return ds[spei_var].chunk({"time": -1})
        except Exception:
            continue
    raise RuntimeError(f"Cannot open: {spei_path}")


def save_results(arrays: dict, lats: np.ndarray, lons: np.ndarray,
                 out_path: Path, model: str, ssp: str,
                 spei_var: str, threshold: float, alpha: float) -> xr.Dataset:
    data_vars = {
        k: (["lat", "lon"], v.astype(np.float32))
        for k, v in arrays.items()
    }
    ds = xr.Dataset(
        data_vars,
        coords={"lat": lats, "lon": lons},
        attrs={
            "title":       "Mann-Kendall Trend Analysis — ISIMIP3b Windows",
            "model":       model,
            "scenario":    ssp,
            "spei_var":    spei_var,
            "threshold":   threshold,
            "mk_method":   "Yue-Wang TFPW (pymannkendall)",
            "alpha":       alpha,
            "windows":     str({k: f"{v['start']}–{v['end']}"
                                for k, v in WINDOWS.items()}),
            "protocol":    "ISIMIP3b",
            "created_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "conventions": "CF-1.8",
        },
    )
    enc = {v: {"zlib": True, "complevel": 4, "dtype": "float32"}
           for v in ds.data_vars}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out_path, encoding=enc)
    print(f"  Saved: {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")
    print(f"  Variables: {len(list(ds.data_vars))}")
    return ds


def print_summary(arrays: dict, lats: np.ndarray, lons: np.ndarray,
                  alpha: float) -> None:
    """Quick significance summary printed after computation."""
    print(f"\n  Significance summary (% cells, p < {alpha}):")
    print(f"  {'Metric':<18} {'ref':>8} {'near':>8} {'far':>8}")
    print(f"  {'-'*46}")
    for metric in METRICS:
        row = f"  {metric:<18}"
        for wk in ["ref", "near", "far"]:
            sig  = arrays[f"{wk}_{metric}_significant"]
            fin  = np.isfinite(arrays[f"{wk}_{metric}_slope"])
            pct  = sig[fin].mean() * 100 if fin.sum() > 0 else 0.0
            row += f" {pct:>7.1f}%"
        print(row)


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Mann-Kendall trend analysis (ISIMIP3b windows) on SPEI-02, all grid cells."
    )
    p.add_argument("--model",      required=True)
    p.add_argument("--ssp",        required=True)
    p.add_argument("--input-dir",  required=True,
                   help="Directory containing spei_{model}_{ssp}.nc files")
    p.add_argument("--out-dir",    required=True)
    p.add_argument("--spei-var",   default="spei_02")
    p.add_argument("--threshold",  default=-1.0,  type=float,
                   help="SPEI drought threshold (default -1.0)")
    p.add_argument("--alpha",      default=0.05,  type=float,
                   help="Significance level (default 0.05)")
    return p.parse_args()


def main():
    args      = parse_args()
    input_dir = Path(args.input_dir)
    out_dir   = Path(args.out_dir)

    spei_path = (input_dir / f"{args.model}_{args.ssp}"
                           / f"spei_{args.model}_{args.ssp}.nc")
    out_path  = out_dir / f"mk_trends_{args.model}_{args.ssp}.nc"

    print(f"\n=== MK Trends: {args.model} / {args.ssp} ===")
    print(f"  Input    : {spei_path}")
    print(f"  Threshold: SPEI < {args.threshold}")
    print(f"  Alpha    : {args.alpha}")

    da = load_spei(spei_path, args.spei_var)
    print(f"  Grid     : {dict(da.sizes)}")
    print(f"  Time     : {str(da.time.values[0])[:10]} → "
          f"{str(da.time.values[-1])[:10]}")

    # Window coverage check
    for wk, w in WINDOWS.items():
        sl   = da.sel(time=slice(w["start"], w["end"]))
        n_yr = sl.sizes["time"] / 12
        flag = "✓" if abs(n_yr - w["n_years"]) < 2 else "⚠ CHECK"
        print(f"  {wk:<5}: {sl.sizes['time']} months ({n_yr:.1f} yr) {flag}")

    print(f"\n  Loading data into memory...")
    da = da.load()

    arrays, lats, lons = run_full_grid(da, args.threshold, args.alpha)

    print_summary(arrays, lats, lons, args.alpha)

    save_results(arrays, lats, lons, out_path,
                 args.model, args.ssp, args.spei_var,
                 args.threshold, args.alpha)

    print(f"\n=== Done: {args.model} / {args.ssp} ===\n")


if __name__ == "__main__":
    main()