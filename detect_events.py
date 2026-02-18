#!/usr/bin/env python3
"""
detect_events.py
Detect drought events across all grid cells and build the event catalog.

Reads:  spei_{model}_{ssp}.nc  +  threshold_p05_monthly_{model}_{ssp}.nc
Output: drought_event_catalog_{model}_{ssp}.nc
  One row per drought event: lat, lon, start_time, end_time,
  duration_months, min_spei, deficit_volume, mean_deficit_rate.

Usage:
  python detect_events.py --model gfdl-esm4 --ssp ssp126 \
      --input-dir /path/to/spei_outputs --out-dir /path/to/results
"""

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import xarray as xr


# =============================================================================
# Core algorithm — one grid cell
# =============================================================================

def detect_events_1d(spei_1d, u_1d, spell_ref):
    """
    Detect drought events in a 1-D time series.

    Spell   : contiguous run where SPEI < spell_ref
    Event   : spell that contains >=1 trigger month (SPEI <= monthly P05)

    Returns (event_id, trigger, min_spei, deficit_contrib) each shape (T,)
    """
    T = spei_1d.shape[0]
    empty = (np.zeros(T, dtype=np.int32), np.zeros(T, dtype=bool),
             np.full(T, np.nan, dtype=np.float32), np.full(T, np.nan, dtype=np.float32))
    if T == 0 or np.all(np.isnan(spei_1d)):
        return empty

    finite    = np.isfinite(spei_1d)
    neg_spell = finite & (spei_1d < spell_ref)
    trigger   = finite & np.isfinite(u_1d) & (spei_1d <= u_1d)

    event_id        = np.zeros(T, dtype=np.int32)
    min_spei        = np.full(T, np.nan, dtype=np.float32)
    deficit_contrib = np.full(T, np.nan, dtype=np.float32)

    neg_idx = np.where(neg_spell)[0]
    if neg_idx.size == 0:
        return event_id, trigger, min_spei, deficit_contrib

    gaps       = np.diff(neg_idx) > 1
    run_starts = np.r_[neg_idx[0],        neg_idx[1:][gaps]]
    run_ends   = np.r_[neg_idx[:-1][gaps], neg_idx[-1]]

    eid = 0
    for s, e in zip(run_starts, run_ends):
        if not np.any(trigger[s:e+1]):
            continue
        eid += 1
        event_id[s:e+1]        = eid
        min_spei[s:e+1]        = np.float32(np.nanmin(spei_1d[s:e+1]))
        contrib                 = (spell_ref - spei_1d[s:e+1]).astype(np.float32)
        deficit_contrib[s:e+1] = np.where(contrib > 0, contrib, 0.0)

    return event_id, trigger, min_spei, deficit_contrib


# =============================================================================
# Loop over all cells
# =============================================================================

def detect_all_cells(spei_np, u_time_np, spell_ref):
    T              = spei_np.shape[0]
    spatial_shape  = spei_np.shape[1:]
    n_cells        = int(np.prod(spatial_shape))

    event_id        = np.zeros((T, n_cells), dtype=np.int32)
    trigger         = np.zeros((T, n_cells), dtype=bool)
    min_spei        = np.full((T, n_cells), np.nan, dtype=np.float32)
    deficit_contrib = np.full((T, n_cells), np.nan, dtype=np.float32)

    for flat in range(n_cells):
        if len(spatial_shape) == 1:
            spei_1d = spei_np[:, flat]
            u_1d    = u_time_np[:, flat]
        else:
            i_lat, i_lon = flat // spatial_shape[1], flat % spatial_shape[1]
            spei_1d = spei_np[:, i_lat, i_lon]
            u_1d    = u_time_np[:, i_lat, i_lon]

        eid, tr, mn, dc         = detect_events_1d(spei_1d, u_1d, spell_ref)
        event_id[:, flat]        = eid
        trigger[:, flat]         = tr
        min_spei[:, flat]        = mn
        deficit_contrib[:, flat] = dc

        if (flat + 1) % 500 == 0:
            print(f"  {flat+1}/{n_cells} cells done")

    return (event_id.reshape(spei_np.shape), trigger.reshape(spei_np.shape),
            min_spei.reshape(spei_np.shape), deficit_contrib.reshape(spei_np.shape))


# =============================================================================
# Build catalog
# =============================================================================

def build_event_catalog(spei, event_id_arr, deficit_arr):
    time_index = pd.to_datetime(spei["time"].values)
    dims       = list(spei.dims)
    spei_np    = np.asarray(spei.values)

    if "cell" in dims:
        lat_vals  = np.atleast_1d(spei["lat"].values)
        lon_vals  = np.atleast_1d(spei["lon"].values)
        cell_iter = [(i, float(lat_vals[i]), float(lon_vals[i]))
                     for i in range(spei.sizes["cell"])]
    else:
        lat_vals  = spei["lat"].values
        lon_vals  = spei["lon"].values
        cell_iter = [(i_lat, i_lon, float(lat), float(lon))
                     for i_lat, lat in enumerate(lat_vals)
                     for i_lon, lon in enumerate(lon_vals)]

    rows: List[Dict[str, Any]] = []
    global_id = 0

    for item in cell_iter:
        if "cell" in dims:
            i_cell, lat, lon = item
            e_1d       = event_id_arr[:, i_cell]
            s_1d       = spei_np[:, i_cell]
            contrib_1d = deficit_arr[:, i_cell]
        else:
            i_lat, i_lon, lat, lon = item
            e_1d       = event_id_arr[:, i_lat, i_lon]
            s_1d       = spei_np[:, i_lat, i_lon]
            contrib_1d = deficit_arr[:, i_lat, i_lon]

        if not np.any(e_1d > 0):
            continue

        for lid in np.unique(e_1d[e_1d > 0]):
            idx         = np.where(e_1d == lid)[0]
            global_id  += 1
            duration    = int(idx.size)
            deficit_vol = float(np.nansum(contrib_1d[idx]))
            rows.append({
                "event":             global_id,
                "lat":               lat,
                "lon":               lon,
                "event_id_local":    int(lid),
                "start_time":        np.datetime64(time_index[idx[0]]),
                "end_time":          np.datetime64(time_index[idx[-1]]),
                "duration_months":   duration,
                "min_spei":          float(np.nanmin(s_1d[idx])),
                "deficit_volume":    deficit_vol,
                "mean_deficit_rate": float(deficit_vol / duration),
            })

    if not rows:
        print("  WARNING: 0 events detected.")
        return _empty_catalog()

    df = pd.DataFrame(rows).sort_values("event")
    ds = xr.Dataset(
        data_vars={
            "lat":               ("event", df["lat"].to_numpy(np.float32)),
            "lon":               ("event", df["lon"].to_numpy(np.float32)),
            "event_id":          ("event", df["event"].to_numpy(np.int32)),
            "event_id_local":    ("event", df["event_id_local"].to_numpy(np.int32)),
            "start_time":        ("event", df["start_time"].to_numpy("datetime64[ns]")),
            "end_time":          ("event", df["end_time"].to_numpy("datetime64[ns]")),
            "duration_months":   ("event", df["duration_months"].to_numpy(np.int32)),
            "min_spei":          ("event", df["min_spei"].to_numpy(np.float32)),
            "deficit_volume":    ("event", df["deficit_volume"].to_numpy(np.float32)),
            "mean_deficit_rate": ("event", df["mean_deficit_rate"].to_numpy(np.float32)),
        },
        coords={"event": df["event"].to_numpy(np.int32)},
        attrs={
            "title": "Drought Event Catalog (per-grid-cell time clustering)",
            "event_definition": (
                "Contiguous run of months with SPEI < spell_ref "
                "containing >=1 trigger month (SPEI <= monthly P05 threshold)."
            ),
            "created_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    )
    ds["lat"].attrs["units"]               = "degrees_north"
    ds["lon"].attrs["units"]               = "degrees_east"
    ds["duration_months"].attrs["units"]   = "months"
    ds["deficit_volume"].attrs["units"]    = "SPEI-month"
    ds["mean_deficit_rate"].attrs["units"] = "SPEI per month"
    return ds


def _empty_catalog():
    empty = lambda dt: ("event", np.array([], dtype=dt))
    return xr.Dataset(
        data_vars={
            "lat":               empty(np.float32),
            "lon":               empty(np.float32),
            "event_id":          empty(np.int32),
            "event_id_local":    empty(np.int32),
            "start_time":        empty("datetime64[ns]"),
            "end_time":          empty("datetime64[ns]"),
            "duration_months":   empty(np.int32),
            "min_spei":          empty(np.float32),
            "deficit_volume":    empty(np.float32),
            "mean_deficit_rate": empty(np.float32),
        },
        coords={"event": np.array([], dtype=np.int32)},
    )


# =============================================================================
# IO
# =============================================================================

def load_spei(spei_path, spei_var, chunks):
    if not spei_path.exists():
        raise FileNotFoundError(f"SPEI file not found: {spei_path}")
    ds = xr.open_dataset(spei_path, chunks=chunks)
    if spei_var not in ds.data_vars:
        raise KeyError(f"Variable '{spei_var}' not in file. Found: {list(ds.data_vars)}")
    return ds[spei_var].chunk({"time": -1})


def load_thresholds(thr_path):
    if not thr_path.exists():
        raise FileNotFoundError(
            f"Threshold file not found: {thr_path}\nRun  compute_thershold.py first."
        )
    ds = xr.open_dataset(thr_path)
    return ds["u"] if "u" in ds.data_vars else ds[list(ds.data_vars)[0]]


def save_catalog(cat, out_path, model, ssp, spei_var, spell_ref, percentile,
                 baseline_start, baseline_end):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cat.attrs.update({
        "model":              model,
        "scenario":           ssp,
        "spei_variable":      spei_var,
        "spell_ref":          spell_ref,
        "threshold_quantile": percentile,
        "baseline_start":     baseline_start,
        "baseline_end":       baseline_end,
    })
    enc = {
        "lat":               {"zlib": True, "complevel": 4, "dtype": "float32"},
        "lon":               {"zlib": True, "complevel": 4, "dtype": "float32"},
        "event_id":          {"zlib": True, "complevel": 4, "dtype": "int32"},
        "event_id_local":    {"zlib": True, "complevel": 4, "dtype": "int32"},
        "start_time":        {"zlib": True, "complevel": 4},
        "end_time":          {"zlib": True, "complevel": 4},
        "duration_months":   {"zlib": True, "complevel": 4, "dtype": "int32"},
        "min_spei":          {"zlib": True, "complevel": 4, "dtype": "float32",
                              "_FillValue": np.float32(np.nan)},
        "deficit_volume":    {"zlib": True, "complevel": 4, "dtype": "float32",
                              "_FillValue": np.float32(np.nan)},
        "mean_deficit_rate": {"zlib": True, "complevel": 4, "dtype": "float32",
                              "_FillValue": np.float32(np.nan)},
    }
    cat.to_netcdf(out_path, encoding=enc)
    print(f"  Saved: {out_path}")


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",          required=True)
    p.add_argument("--ssp",            required=True)
    p.add_argument("--input-dir",      required=True)
    p.add_argument("--out-dir",        required=True)
    p.add_argument("--spei-var",       default="spei_02")
    p.add_argument("--baseline-start", default="1979-01-01")
    p.add_argument("--baseline-end",   default="2014-12-31")
    p.add_argument("--percentile",     default=0.05, type=float)
    p.add_argument("--spell-ref",      default=-0.4,  type=float)
    return p.parse_args()


def main():
    args      = parse_args()
    input_dir = Path(args.input_dir)
    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    spei_path = input_dir / f"{args.model}_{args.ssp}" / f"spei_{args.model}_{args.ssp}.nc"
    thr_path  = out_dir   / f"threshold_p{int(args.percentile*100):02d}_monthly_{args.model}_{args.ssp}.nc"
    cat_path  = out_dir   / f"drought_event_catalog_{args.model}_{args.ssp}.nc"
    chunks    = {"time": -1, "lat": 25, "lon": 25}

    print(f"\n=== Events: {args.model} / {args.ssp} ===")
    spei = load_spei(spei_path, args.spei_var, chunks)
    u    = load_thresholds(thr_path)

    print(f"  Grid: {dict(spei.sizes)}")

    # Map u(month, ...) → u_time(time, ...) for every timestep
    mon    = spei["time"].dt.month
    u_time = u.sel(month=mon)
    u_time = u_time.transpose("time", *[d for d in u_time.dims if d != "time"])
    spei, u_time = xr.align(spei, u_time, join="exact")

    spei_np   = np.asarray(spei.load().values)
    u_time_np = np.asarray(u_time.load().values)

    n_cells = int(np.prod(spei_np.shape[1:]))
    print(f"  Running event detection over {n_cells} cells...")
    event_id_arr, _, _, deficit_arr = detect_all_cells(spei_np, u_time_np, args.spell_ref)

    print("  Building event catalog...")
    cat = build_event_catalog(spei, event_id_arr, deficit_arr)

    n_events = int(cat.sizes.get("event", 0))
    print(f"  Events detected: {n_events}")
    if n_events > 0:
        dur = cat["duration_months"].values
        print(f"  Duration: min={dur.min()}, median={np.median(dur):.1f}, max={dur.max()}")

    save_catalog(cat, cat_path, args.model, args.ssp, args.spei_var,
                 args.spell_ref, args.percentile, args.baseline_start, args.baseline_end)


if __name__ == "__main__":
    main()