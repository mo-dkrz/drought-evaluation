#!/usr/bin/env python3
"""
compute_thershold.py
Compute monthly 5th-percentile SPEI thresholds over the baseline period.

Output: {out_dir}/threshold_p05_monthly_{model}_{ssp}.nc
  Variable u(month, lat, lon)

Usage:
  python compute_thershold.py --model gfdl-esm4 --ssp ssp126 \
      --input-dir /path/to/spei_outputs --out-dir /path/to/results
"""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr


def compute_monthly_thresholds(spei, baseline_start, baseline_end, percentile):
    base = spei.sel(time=slice(baseline_start, baseline_end))
    if base.time.size == 0:
        raise ValueError(
            f"Baseline {baseline_start}–{baseline_end} returned 0 timesteps. "
            f"Data covers {str(spei.time.values[0])[:10]} – {str(spei.time.values[-1])[:10]}."
        )
    u = base.groupby("time.month").quantile(percentile, dim="time").rename("u")
    u.attrs.update({
        "long_name":      f"Baseline monthly {percentile:.0%} quantile threshold",
        "baseline_start": baseline_start,
        "baseline_end":   baseline_end,
        "quantile":       percentile,
    })
    return u


def save_thresholds(u, out_path, model, ssp, spei_var, baseline_start, baseline_end, percentile):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds = xr.Dataset({"u": u}, attrs={
        "title":          "Monthly drought trigger thresholds (baseline P05)",
        "model":          model,
        "scenario":       ssp,
        "spei_variable":  spei_var,
        "baseline_start": baseline_start,
        "baseline_end":   baseline_end,
        "created_utc":    datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "conventions":    "CF-1.8",
    })
    enc = {"u": {"zlib": True, "complevel": 4, "dtype": "float32", "_FillValue": np.float32(np.nan)}}
    ds.to_netcdf(out_path, encoding=enc)
    print(f"  Saved: {out_path}")


def load_spei(spei_path, spei_var, chunks):
    if not spei_path.exists():
        raise FileNotFoundError(f"SPEI file not found: {spei_path}")
    ds = xr.open_dataset(spei_path, chunks=chunks)
    if spei_var not in ds.data_vars:
        raise KeyError(f"Variable '{spei_var}' not in file. Found: {list(ds.data_vars)}")
    return ds[spei_var].chunk({"time": -1})


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
    return p.parse_args()


def main():
    args      = parse_args()
    input_dir = Path(args.input_dir)
    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    spei_path = input_dir / f"{args.model}_{args.ssp}" / f"spei_{args.model}_{args.ssp}.nc"
    out_path  = out_dir   / f"threshold_p{int(args.percentile*100):02d}_monthly_{args.model}_{args.ssp}.nc"

    print(f"\n=== Thresholds: {args.model} / {args.ssp} ===")
    spei = load_spei(spei_path, args.spei_var, chunks={"time": -1, "lat": 25, "lon": 25})
    print(f"  Grid : {dict(spei.sizes)}")
    print(f"  Time : {str(spei.time.values[0])[:10]} – {str(spei.time.values[-1])[:10]}")

    u = compute_monthly_thresholds(spei, args.baseline_start, args.baseline_end, args.percentile)
    print(f"  Shape: {u.dims} {u.shape}")
    print(f"  Range: {float(u.min()):.3f} to {float(u.max()):.3f}")

    save_thresholds(u, out_path, args.model, args.ssp, args.spei_var,
                    args.baseline_start, args.baseline_end, args.percentile)


if __name__ == "__main__":
    main()