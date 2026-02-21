#!/usr/bin/env python3
"""
compute_return_periods.py
POT/GPD return period analysis for ALL grid cells in the drought event catalog.

Reads:  drought_event_catalog_{model}_{ssp}.nc
Output: return_periods_{model}_{ssp}.nc

  Dimensions: cell, return_period
  Variables (per variable=intensity/severity, per slice=hist/far):
    lat, lon
    gpd_shape_{var}_{slice}        GPD shape parameter (ξ)
    gpd_scale_{var}_{slice}        GPD scale parameter (σ)
    lambda_{var}_{slice}           Annual event rate (λ)
    n_events_{var}_{slice}         Event count in slice
    ks_pval_{var}_{slice}          KS test p-value
    rp_quantile_{var}_{slice}      (cell × return_period) quantile values
    delta_sev_abs_{var}            Metric A: absolute severity change at each RP
    delta_sev_rel_{var}            Metric A: relative severity change (%)
    rp_far_equiv_{var}             Metric B: future RP equivalent to hist threshold
    rp_ratio_{var}                 Metric B: hist_RP / far_RP
    delta_lambda_{var}_abs         Metric C: absolute λ change
    delta_lambda_{var}_rel         Metric C: relative λ change (%)

Usage:
  python compute_return_periods.py --model gfdl-esm4 --ssp ssp126 \
      --out-dir ~/drought_catalog
"""

import argparse
import warnings
from datetime import datetime
from pathlib import Path

import multiprocessing as mp
import time
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import genpareto, kstest

# =============================================================================
# Defaults (mirrors batch_drought.sh)
# =============================================================================
DEFAULT_SLICES = {
    "hist": ("1950-01-01", "2014-12-31"),
    "far":  ("2036-01-01", "2100-12-31"),
}
DEFAULT_RETURN_PERIODS = [2, 5, 10, 20, 50, 100]
MIN_EVENTS = 10   # minimum for a GPD fit attempt


# =============================================================================
# Core statistics
# =============================================================================

def compute_lambda(df_slice: pd.DataFrame, start: str, end: str):
    n_years  = pd.Timestamp(end).year - pd.Timestamp(start).year + 1
    n_events = len(df_slice)
    return n_events / n_years, n_events, n_years


def fit_gpd(values: np.ndarray, lam: float, return_periods: list) -> dict:
    """Fit GPD to POT exceedances and compute RP quantiles."""
    out = {
        "shape": np.nan, "scale": np.nan,
        "ks_stat": np.nan, "ks_pval": np.nan,
        "n": len(values), "lam": lam,
        "converged": False,
        "rp_quantiles": {rp: np.nan for rp in return_periods},
    }
    if len(values) < MIN_EVENTS:
        return out
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            shape, loc, scale = genpareto.fit(values, floc=0)
        out["shape"] = shape
        out["scale"] = scale
        out["converged"] = True
        ks_stat, ks_pval = kstest(values, "genpareto", args=(shape, loc, scale))
        out["ks_stat"] = ks_stat
        out["ks_pval"] = ks_pval
        for rp in return_periods:
            p_annual = 1.0 / rp
            p_event  = 1.0 - np.exp(-p_annual / lam)
            p_event  = np.clip(p_event, 1e-9, 1 - 1e-9)
            q = genpareto.ppf(1 - p_event, shape, loc=0, scale=scale)
            out["rp_quantiles"][rp] = float(q) if np.isfinite(q) else np.nan
    except Exception:
        pass
    return out


def gpd_quantile(shape, scale, lam, rp):
    """Return the GPD quantile for an annual return period."""
    try:
        p_event = np.clip(1.0 - np.exp(-1.0 / (rp * lam)), 1e-9, 1 - 1e-9)
        q = genpareto.ppf(1 - p_event, shape, loc=0, scale=scale)
        return float(q) if np.isfinite(q) and q > 0 else np.nan
    except Exception:
        return np.nan


def gpd_return_period(shape, scale, lam, threshold):
    """Return the annual RP for a given severity threshold."""
    try:
        p_exc = genpareto.sf(threshold, shape, loc=0, scale=scale)
        if p_exc <= 0:
            return np.inf
        rp = 1.0 / (lam * p_exc)
        return float(rp) if np.isfinite(rp) else np.nan
    except Exception:
        return np.nan


# =============================================================================
# Process one grid cell
# =============================================================================

def process_cell(df_cell: pd.DataFrame, slices: dict,
                 return_periods: list) -> dict:
    """
    Run GPD analysis for one (lat, lon) cell across all slices and both variables.
    Returns a flat dict of scalar results and per-RP arrays.
    """
    result = {}

    for var, col, negate in [("intensity", "min_spei",       True),
                               ("severity",  "deficit_volume", False)]:
        fits = {}
        for sl_name, (sl_start, sl_end) in slices.items():
            mask     = ((df_cell["start_time"] >= sl_start) &
                        (df_cell["start_time"] <= sl_end))
            df_sl    = df_cell[mask]
            vals     = df_sl[col].values.astype(float)
            if negate:
                vals = -vals
            vals = vals[np.isfinite(vals)]

            lam, n_events, _ = compute_lambda(df_sl, sl_start, sl_end)
            fit = fit_gpd(vals, lam, return_periods)
            fits[sl_name] = fit

            result[f"shape_{var}_{sl_name}"]   = fit["shape"]
            result[f"scale_{var}_{sl_name}"]   = fit["scale"]
            result[f"lambda_{var}_{sl_name}"]  = lam
            result[f"n_events_{var}_{sl_name}"]= n_events
            result[f"ks_pval_{var}_{sl_name}"] = fit["ks_pval"]
            result[f"rp_quantiles_{var}_{sl_name}"] = np.array(
                [fit["rp_quantiles"].get(rp, np.nan) for rp in return_periods],
                dtype=np.float32)

        # Climate metrics (hist vs far)
        for rp in return_periods:
            f_h = fits.get("hist", {})
            f_f = fits.get("far",  {})
            if not f_h.get("converged") or not f_f.get("converged"):
                result[f"delta_sev_abs_{var}_{rp}"] = np.nan
                result[f"delta_sev_rel_{var}_{rp}"] = np.nan
                result[f"rp_far_equiv_{var}_{rp}"]  = np.nan
                result[f"rp_ratio_{var}_{rp}"]      = np.nan
            else:
                q_h = gpd_quantile(f_h["shape"], f_h["scale"], f_h["lam"], rp)
                q_f = gpd_quantile(f_f["shape"], f_f["scale"], f_f["lam"], rp)
                d_abs = (q_f - q_h) if (np.isfinite(q_h) and np.isfinite(q_f)) else np.nan
                d_rel = (d_abs / q_h * 100) if (np.isfinite(d_abs) and q_h > 0) else np.nan
                result[f"delta_sev_abs_{var}_{rp}"] = d_abs
                result[f"delta_sev_rel_{var}_{rp}"] = d_rel

                rp_far = (gpd_return_period(f_f["shape"], f_f["scale"],
                                             f_f["lam"], q_h)
                          if np.isfinite(q_h) and q_h > 0 else np.nan)
                rp_ratio = (rp / rp_far
                            if (np.isfinite(rp_far) and rp_far > 0) else np.nan)
                result[f"rp_far_equiv_{var}_{rp}"] = rp_far
                result[f"rp_ratio_{var}_{rp}"]     = rp_ratio

        # Metric C: lambda change (scalar per variable)
        lam_h = fits.get("hist", {}).get("lam", np.nan)
        lam_f = fits.get("far",  {}).get("lam", np.nan)
        dl    = (lam_f - lam_h) if (np.isfinite(lam_h) and np.isfinite(lam_f)) else np.nan
        result[f"delta_lambda_{var}_abs"] = dl
        result[f"delta_lambda_{var}_rel"] = (dl / lam_h * 100
                                              if np.isfinite(dl) and lam_h > 0 else np.nan)
    return result


# =============================================================================
# Assemble output NetCDF
# =============================================================================

def build_output_dataset(all_results: list, lats: np.ndarray, lons: np.ndarray,
                          slices: dict, return_periods: list,
                          model: str, ssp: str) -> xr.Dataset:

    n_cells = len(all_results)
    n_rp    = len(return_periods)
    rp_arr  = np.array(return_periods, dtype=np.int32)

    data_vars = {
        "lat": (["cell"], lats.astype(np.float32),
                {"units": "degrees_north"}),
        "lon": (["cell"], lons.astype(np.float32),
                {"units": "degrees_east"}),
    }

    for var in ("intensity", "severity"):
        for sl in slices:
            for scalar_key, long_name in [
                (f"shape_{var}_{sl}",    f"GPD shape ξ ({var}, {sl})"),
                (f"scale_{var}_{sl}",    f"GPD scale σ ({var}, {sl})"),
                (f"lambda_{var}_{sl}",   f"Annual event rate λ ({var}, {sl})"),
                (f"n_events_{var}_{sl}", f"Event count ({var}, {sl})"),
                (f"ks_pval_{var}_{sl}",  f"KS p-value ({var}, {sl})"),
            ]:
                arr = np.array([r.get(scalar_key, np.nan) for r in all_results],
                               dtype=np.float32)
                data_vars[scalar_key] = (["cell"], arr, {"long_name": long_name})

            # RP quantile matrix
            qkey   = f"rp_quantiles_{var}_{sl}"
            qmat   = np.array([r.get(qkey, np.full(n_rp, np.nan))
                                for r in all_results], dtype=np.float32)
            data_vars[f"rp_quantile_{var}_{sl}"] = (
                ["cell", "return_period"], qmat,
                {"long_name": f"GPD quantile ({var}, {sl})",
                 "units":     "SPEI-months" if var == "severity" else "-SPEI"})

        # Climate metrics
        for rp in return_periods:
            for mkey, lname in [
                (f"delta_sev_abs_{var}_{rp}", f"Severity Δ abs at RP={rp} ({var})"),
                (f"delta_sev_rel_{var}_{rp}", f"Severity Δ % at RP={rp} ({var})"),
                (f"rp_far_equiv_{var}_{rp}",  f"Equivalent future RP at hist RP={rp} ({var})"),
                (f"rp_ratio_{var}_{rp}",      f"RP ratio hist/far at RP={rp} ({var})"),
            ]:
                arr = np.array([r.get(mkey, np.nan) for r in all_results],
                               dtype=np.float32)
                data_vars[mkey] = (["cell"], arr, {"long_name": lname})

        for mkey, lname in [
            (f"delta_lambda_{var}_abs", f"Δλ absolute ({var})"),
            (f"delta_lambda_{var}_rel", f"Δλ relative % ({var})"),
        ]:
            arr = np.array([r.get(mkey, np.nan) for r in all_results],
                           dtype=np.float32)
            data_vars[mkey] = (["cell"], arr, {"long_name": lname})

    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "cell":          np.arange(n_cells, dtype=np.int32),
            "return_period": rp_arr,
        },
        attrs={
            "title":          "Drought Return Period Analysis (all grid cells)",
            "model":          model,
            "scenario":       ssp,
            "slices":         str({k: v for k, v in slices.items()}),
            "return_periods": str(return_periods),
            "min_events":     MIN_EVENTS,
            "created_utc":    datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "conventions":    "CF-1.8",
        },
    )
    return ds


# =============================================================================
# Main
# =============================================================================

def _cell_worker(lat_r, lon_r, df_cell, slices, return_periods):
    """Module-level worker for multiprocessing (must be picklable)."""
    return lat_r, lon_r, process_cell(df_cell, slices, return_periods)




def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",          required=True)
    p.add_argument("--ssp",            required=True)
    p.add_argument("--catalog-dir",    required=True,
                   help="Directory containing drought_event_catalog_*.nc "
                        "(i.e. --out-dir from detect_events.py)")
    p.add_argument("--out-dir",        required=True,
                   help="Directory to write return_periods_*.nc")
    p.add_argument("--return-periods", default="2,5,10,20,50,100",
                   help="Comma-separated list of return periods in years")
    p.add_argument("--hist-start",     default="1950-01-01")
    p.add_argument("--hist-end",       default="2014-12-31")
    p.add_argument("--far-start",      default="2036-01-01")
    p.add_argument("--far-end",        default="2100-12-31")
    return p.parse_args()


def main():
    args        = parse_args()
    catalog_dir = Path(args.catalog_dir)
    out_dir     = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    slices = {
        "hist": (args.hist_start, args.hist_end),
        "far":  (args.far_start,  args.far_end),
    }
    return_periods = [int(x) for x in args.return_periods.split(",")]

    catalog_path = catalog_dir / f"drought_event_catalog_{args.model}_{args.ssp}.nc"
    out_path     = out_dir     / f"return_periods_{args.model}_{args.ssp}.nc"

    print(f"\n=== Return Periods: {args.model} / {args.ssp} ===")
    print(f"  Catalog : {catalog_path}")

    if not catalog_path.exists():
        raise FileNotFoundError(
            f"Catalog not found: {catalog_path}\nRun detect_events.py first.")

    # ── Load catalog ──────────────────────────────────────────────────────────
    ds  = xr.open_dataset(catalog_path)
    df  = pd.DataFrame({
        "lat":            ds["lat"].values,
        "lon":            ds["lon"].values,
        "start_time":     pd.to_datetime(ds["start_time"].values),
        "min_spei":       ds["min_spei"].values,
        "deficit_volume": ds["deficit_volume"].values,
    })
    ds.close()
    print(f"  Events  : {len(df):,}")
    print(f"  Time    : {df['start_time'].min().date()} – {df['start_time'].max().date()}")

    # ── Group by unique (lat, lon) ────────────────────────────────────────────
    # Round coords to avoid floating-point duplicates
    df["lat_r"] = df["lat"].round(6)
    df["lon_r"] = df["lon"].round(6)
    cells = df.groupby(["lat_r", "lon_r"], sort=False)
    n_cells = cells.ngroups
    print(f"  Cells   : {n_cells:,}")
    print(f"  Slices  : { {k: v for k, v in slices.items()} }")
    print(f"  RPs     : {return_periods}")
    print(f"  Running…")

    # Build list of (lat, lon, df_cell) tuples for parallel processing
    cell_list = [(lat_r, lon_r, df_cell.copy(), slices, return_periods)
                 for (lat_r, lon_r), df_cell in cells]

    n_workers = min(mp.cpu_count(), n_cells)
    print(f"  Workers : {n_workers}")

    t0 = time.time()
    all_results, lats_out, lons_out = [], [], []

    with mp.Pool(n_workers) as pool:
        for i, (lat_r, lon_r, result) in enumerate(
                pool.starmap(_cell_worker, cell_list, chunksize=50), 1):
            all_results.append(result)
            lats_out.append(lat_r)
            lons_out.append(lon_r)
            if i % 500 == 0 or i == n_cells:
                elapsed = time.time() - t0
                rate    = i / elapsed
                eta     = (n_cells - i) / rate if rate > 0 else 0
                print(f"  {i:>6}/{n_cells}  "
                      f"{elapsed:.0f}s elapsed  ETA {eta:.0f}s", flush=True)

    print(f"  {n_cells}/{n_cells} cells complete  "
          f"({time.time() - t0:.1f}s total)")

    # ── Build & save dataset ──────────────────────────────────────────────────
    ds_out = build_output_dataset(
        all_results,
        np.array(lats_out, dtype=np.float32),
        np.array(lons_out, dtype=np.float32),
        slices, return_periods, args.model, args.ssp,
    )

    enc = {v: {"zlib": True, "complevel": 4}
           for v in ds_out.data_vars
           if ds_out[v].dtype in (np.float32, np.int32)}
    ds_out.to_netcdf(out_path, encoding=enc)
    print(f"  Saved : {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")

    # ── Quick summary ─────────────────────────────────────────────────────────
    for var in ("intensity", "severity"):
        for sl in slices:
            n_fit = int(
                np.sum(np.isfinite(ds_out[f"shape_{var}_{sl}"].values)))
            print(f"  Cells with GPD fit  [{var:>9} / {sl}]: {n_fit:>6}/{n_cells}")

    print(f"\n=== Done: {args.model} / {args.ssp} ===\n")


if __name__ == "__main__":
    main()