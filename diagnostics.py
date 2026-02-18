#!/usr/bin/env python3
"""
03_diagnostics.py
Diagnostic plots for the drought pipeline outputs.

Reads the outputs of 01 and 02, plus the original SPEI file.
Produces three sets of figures:

  1. threshold/  — spatial distribution of monthly P05 thresholds
                   (boxplot by month + histogram grid)

  2. timeseries/ — SPEI curve + P05 threshold + triggered months +
                   deficit shading + event minima for sample locations
                   (full period + zoomed window)

  3. catalog/    — event catalog distributions
                   (duration, deficit volume, min SPEI histograms)

Usage:
  python 03_diagnostics.py --model gfdl-esm4 --ssp ssp126 \
      --input-dir /path/to/spei_outputs \
      --out-dir   /path/to/drought_catalog

  # Custom sample locations and zoom window:
  python 03_diagnostics.py --model gfdl-esm4 --ssp ssp126 \
      --input-dir /path/to/spei_outputs \
      --out-dir   /path/to/drought_catalog \
      --locations "Atlanta:33.7:-84.4" "Seattle:47.6:-122.3" \
      --zoom-start 1980-01-01 --zoom-end 1999-12-31
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import xarray as xr
import os

try:
    import cartopy
    cartopy.config["data_dir"] = os.path.join(os.environ["HOME"], "cartopy_data")
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    print("WARNING: cartopy not installed — heatmaps will be plotted without boundaries.")


# =============================================================================
# 1. Threshold diagnostics
# =============================================================================

def plot_threshold_diagnostics(thr_path: Path, fig_dir: Path, spell_ref: float) -> None:
    """
    Boxplot + histogram grid of monthly P05 threshold spatial distribution.
    """
    fig_dir.mkdir(parents=True, exist_ok=True)
    thr = xr.open_dataset(thr_path)
    u   = thr["u"] if "u" in thr.data_vars else thr[list(thr.data_vars)[0]]

    # Drop quantile coord if xarray added it
    for c in list(u.coords):
        if c not in u.dims:
            u = u.drop_vars(c)

    month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"]

    # Flatten spatial dims per month
    spatial_dims = [d for d in u.dims if d != "month"]
    u_flat = {m: u.sel(month=m).values.ravel() for m in range(1, 13)}
    u_flat = {m: v[np.isfinite(v)] for m, v in u_flat.items()}

    # --- Boxplot ---
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.boxplot([u_flat[m] for m in range(1, 13)], showfliers=False,
               medianprops=dict(color="orange", linewidth=1.5))
    ax.axhline(spell_ref, linestyle="--", lw=0.9, alpha=0.6,
               label=f"spell_ref ({spell_ref})")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(month_labels)
    ax.set_ylabel("SPEI threshold (5th percentile)")
    ax.set_title(f"Spatial distribution of monthly P05 thresholds  —  {thr_path.stem}")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = fig_dir / "threshold_boxplot_by_month.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")

    # --- Histogram grid (12 panels) ---
    fig, axes = plt.subplots(3, 4, figsize=(12, 7), sharex=True, sharey=True)
    for i, ax in enumerate(axes.ravel()):
        ax.hist(u_flat[i+1], bins=40)
        ax.set_title(month_labels[i], fontsize=10)
        ax.grid(True, alpha=0.25)
        if i % 4 == 0:
            ax.set_ylabel("Count")
        if i >= 8:
            ax.set_xlabel("Threshold")
    fig.suptitle(f"Monthly P05 threshold histograms  —  {thr_path.stem}")
    fig.tight_layout()
    out = fig_dir / "threshold_histograms_12months.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")

    thr.close()


# =============================================================================
# 2. Point-level time series
# =============================================================================

def plot_timeseries(spei_path: Path, thr_path: Path, fig_dir: Path,
                    spei_var: str, spell_ref: float,
                    locations: dict,
                    zoom: tuple) -> None:
    """
    For each sample location: full-period + zoomed SPEI time series with
    P05 threshold, triggered months, deficit shading, event minima.
    """
    fig_dir.mkdir(parents=True, exist_ok=True)

    ds_spei = xr.open_dataset(spei_path)
    ds_thr  = xr.open_dataset(thr_path)
    u       = ds_thr["u"] if "u" in ds_thr.data_vars else ds_thr[list(ds_thr.data_vars)[0]]
    for c in list(u.coords):
        if c not in u.dims:
            u = u.drop_vars(c)

    spei_da     = ds_spei[spei_var]
    t           = pd.to_datetime(spei_da["time"].values)
    month_index = spei_da["time"].dt.month

    for name, (lat, lon) in locations.items():
        print(f"  {name}...")
        spei_1d   = np.asarray(spei_da.sel(lat=lat, lon=lon, method="nearest").values, dtype=np.float64)
        u_time_1d = np.asarray(u.sel(month=month_index)
                                .sel(lat=lat, lon=lon, method="nearest").values, dtype=np.float64)

        # Re-detect events for this cell (needed for triggers + deficit)
        event_id_1d, trigger_1d, deficit_contrib_1d = _detect_1d(spei_1d, u_time_1d, spell_ref)

        for mode in ("full", "zoom"):
            if mode == "zoom":
                mask = (t >= pd.Timestamp(zoom[0])) & (t <= pd.Timestamp(zoom[1]))
                t_plot   = t[mask]
                spei_plot  = spei_1d[mask]
                u_plot     = u_time_1d[mask]
                trig_plot  = trigger_1d[mask]
                def_plot   = deficit_contrib_1d[mask]
                eid_plot   = event_id_1d[mask]
                suffix     = f"zoom_{zoom[0][:4]}_{zoom[1][:4]}"
            else:
                t_plot   = t
                spei_plot  = spei_1d
                u_plot     = u_time_1d
                trig_plot  = trigger_1d
                def_plot   = deficit_contrib_1d
                eid_plot   = event_id_1d
                suffix     = "full"

            # Event minimum points (one red dot per event)
            min_times, min_vals = [], []
            for eid in np.unique(eid_plot[eid_plot > 0]):
                idx   = np.where(eid_plot == eid)[0]
                imin  = idx[np.argmin(spei_plot[idx])]
                min_times.append(t_plot[imin])
                min_vals.append(float(spei_plot[imin]))

            figw = 16 if mode == "full" else 14
            fig, ax = plt.subplots(figsize=(figw, 5))

            ax.plot(t_plot, spei_plot, lw=0.9, color="C0", label="SPEI-02")
            ax.plot(t_plot, u_plot,    lw=0.9, color="C1", label="Monthly P05 threshold")
            ax.axhline(spell_ref, linestyle="--", lw=0.8, color="lightblue",
                       label=f"Spell ref ({spell_ref})")
            ax.axhline(0, lw=0.6, alpha=0.4)

            # Deficit shading (orange area between SPEI and spell_ref)
            shade = np.where(np.isfinite(def_plot), def_plot, 0) > 0
            if shade.any():
                ax.fill_between(t_plot, spei_plot,
                                np.full_like(spei_plot, spell_ref),
                                where=shade, interpolate=True,
                                alpha=0.3, color="orange", label="Deficit")

            # Triggered months
            trig_t = t_plot[trig_plot]
            trig_v = spei_plot[trig_plot]
            if len(trig_t):
                ax.scatter(trig_t, trig_v, s=30 if mode == "full" else 60,
                           color="darkblue", edgecolors="black", zorder=4,
                           label="Triggered months")

            # Event minima
            if min_times:
                ax.scatter(min_times, min_vals, s=60, color="red",
                           edgecolors="darkred", zorder=5, label="Event minimum")

            ax.set_title(f"{name} — {suffix}")
            ax.set_ylabel("SPEI-02")
            ax.grid(True, linestyle=":", alpha=0.4)
            ax.legend(loc="lower left", frameon=True, fontsize=8)

            if mode == "full":
                ax.xaxis.set_major_locator(mdates.YearLocator(base=20))
            else:
                ax.xaxis.set_major_locator(mdates.YearLocator(base=2))
                ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=(1, 7)))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            fig.tight_layout()

            out = fig_dir / f"{name.replace(' ','_')}_{suffix}.png"
            fig.savefig(out, dpi=150)
            plt.close(fig)

    ds_spei.close()
    ds_thr.close()
    print(f"  Figures saved to: {fig_dir}")


def _detect_1d(spei_1d, u_1d, spell_ref):
    """Minimal 1-D event detection (returns event_id, trigger, deficit_contrib)."""
    T         = len(spei_1d)
    event_id  = np.zeros(T, dtype=np.int32)
    trigger   = np.isfinite(spei_1d) & np.isfinite(u_1d) & (spei_1d <= u_1d)
    deficit   = np.full(T, np.nan, dtype=np.float32)

    neg_idx = np.where(np.isfinite(spei_1d) & (spei_1d < spell_ref))[0]
    if neg_idx.size == 0:
        return event_id, trigger, deficit

    gaps   = np.diff(neg_idx) > 1
    starts = np.r_[neg_idx[0],        neg_idx[1:][gaps]]
    ends   = np.r_[neg_idx[:-1][gaps], neg_idx[-1]]

    eid = 0
    for s, e in zip(starts, ends):
        if not np.any(trigger[s:e+1]):
            continue
        eid += 1
        event_id[s:e+1] = eid
        contrib = (spell_ref - spei_1d[s:e+1]).astype(np.float32)
        deficit[s:e+1]  = np.where(contrib > 0, contrib, 0.0)

    return event_id, trigger, deficit


# =============================================================================
# 3. Heatmaps
# =============================================================================

def plot_heatmaps(cat_path: Path, thr_path: Path, fig_dir: Path) -> None:
    """
    Spatial heatmaps (lat x lon) aggregated from the event catalog:
      - event count per grid cell
      - mean duration
      - mean deficit volume
      - mean min SPEI
    """
    fig_dir.mkdir(parents=True, exist_ok=True)

    cat = xr.open_dataset(cat_path)
    thr = xr.open_dataset(thr_path)
    u   = thr["u"] if "u" in thr.data_vars else thr[list(thr.data_vars)[0]]
    lat_vals = u["lat"].values
    lon_vals = u["lon"].values

    if cat.sizes["event"] == 0:
        print("  No events — skipping heatmaps.")
        cat.close(); thr.close()
        return

    # Aggregate catalog per grid cell
    df = cat[["lat", "lon", "duration_months", "deficit_volume", "min_spei"]].to_dataframe()
    grouped = df.groupby(["lat", "lon"]).agg(
        event_count    =("duration_months", "count"),
        mean_duration  =("duration_months", "mean"),
        mean_deficit   =("deficit_volume",  "mean"),
        mean_min_spei  =("min_spei",        "mean"),
    ).reset_index()

    # Build 2-D grids
    lat_idx = {v: i for i, v in enumerate(lat_vals)}
    lon_idx = {v: i for i, v in enumerate(lon_vals)}

    metrics = {
        "event_count":   ("Event count",              "YlOrRd"),
        "mean_duration": ("Mean duration (months)",   "YlOrRd"),
        "mean_deficit":  ("Mean deficit volume",      "YlOrRd"),
        "mean_min_spei": ("Mean min SPEI",            "RdBu"),
    }

    for col, (title, cmap) in metrics.items():
        grid = np.full((len(lat_vals), len(lon_vals)), np.nan)
        for _, row in grouped.iterrows():
            i = lat_idx.get(row["lat"])
            j = lon_idx.get(row["lon"])
            if i is not None and j is not None:
                grid[i, j] = row[col]

        if HAS_CARTOPY:
            fig, ax = plt.subplots(figsize=(10, 5),
                                   subplot_kw={"projection": ccrs.PlateCarree()})
            im = ax.pcolormesh(lon_vals, lat_vals, grid, cmap=cmap,
                               shading="auto", transform=ccrs.PlateCarree())
            ax.add_feature(cfeature.STATES, linewidth=0.5, edgecolor="gray")
            ax.add_feature(cfeature.BORDERS, linewidth=0.8, edgecolor="black")
            ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
            ax.set_extent([lon_vals.min(), lon_vals.max(),
                           lat_vals.min(), lat_vals.max()], crs=ccrs.PlateCarree())
        else:
            fig, ax = plt.subplots(figsize=(10, 5))
            im = ax.pcolormesh(lon_vals, lat_vals, grid, cmap=cmap, shading="auto")
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")

        plt.colorbar(im, ax=ax, shrink=0.7)
        ax.set_title(f"{title}  —  {cat_path.stem}")
        fig.tight_layout()
        out = fig_dir / f"heatmap_{col}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  Saved: {out}")

    cat.close(); thr.close()


# =============================================================================
# 4. Catalog distributions
# =============================================================================

def plot_catalog_distributions(cat_path: Path, fig_dir: Path) -> None:
    """
    Histograms of event duration, deficit volume, and min SPEI from the catalog.
    """
    fig_dir.mkdir(parents=True, exist_ok=True)
    cat = xr.open_dataset(cat_path)
    n   = int(cat.sizes["event"])

    if n == 0:
        print("  No events in catalog — skipping distribution plots.")
        cat.close()
        return

    dur  = cat["duration_months"].values
    dv   = cat["deficit_volume"].values
    mn   = cat["min_spei"].values
    mdr  = cat["mean_deficit_rate"].values

    print(f"  Events: {n}")
    print(f"  Duration   : min={dur.min()}, median={np.median(dur):.1f}, max={dur.max()}")
    print(f"  Deficit vol: min={np.nanmin(dv):.3f}, median={np.nanmedian(dv):.3f}, max={np.nanmax(dv):.3f}")
    print(f"  Min SPEI   : min={np.nanmin(mn):.3f}, median={np.nanmedian(mn):.3f}")

    specs = [
        (dur,                       "Duration (months)",                  "duration_months"),
        (dv,                        "Deficit volume (SPEI-month)",        "deficit_volume"),
        (mn,                        "Minimum SPEI within event",          "min_spei"),
        (mdr,                       "Mean deficit rate (SPEI per month)", "mean_deficit_rate"),
    ]

    for data, xlabel, fname in specs:
        lo, hi = np.nanpercentile(data, [1, 99])
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(data, bins=60)
        ax.set_xlim(lo, hi)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count of events")
        ax.set_title(f"{xlabel}  —  {cat_path.stem}\n(n={n}, 1st–99th pct shown)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out = fig_dir / f"dist_{fname}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  Saved: {out}")

    cat.close()


# =============================================================================
# CLI
# =============================================================================

def parse_locations(loc_strings):
    """Parse 'Name:lat:lon' strings into {name: (lat, lon)} dict."""
    result = {}
    for s in loc_strings:
        parts = s.split(":")
        if len(parts) != 3:
            raise ValueError(f"Location must be 'Name:lat:lon', got: {s}")
        result[parts[0]] = (float(parts[1]), float(parts[2]))
    return result


DEFAULT_LOCATIONS = [
    "Atlanta:33.7:-84.4",
    "Lawrence:39.0:-95.2",
    "Phoenix:33.4:-112.0",
    "Seattle:47.6:-122.3",
    "Ithaca:42.4:-76.5",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",       required=True)
    p.add_argument("--ssp",         required=True)
    p.add_argument("--input-dir",   required=True)
    p.add_argument("--out-dir",     required=True)
    p.add_argument("--spei-var",    default="spei_02")
    p.add_argument("--percentile",  default=0.05, type=float)
    p.add_argument("--spell-ref",   default=-0.4,  type=float)
    p.add_argument("--locations",   nargs="+", default=DEFAULT_LOCATIONS,
                   metavar="Name:lat:lon",
                   help="Sample locations for time-series plots")
    p.add_argument("--zoom-start",  default="1980-01-01")
    p.add_argument("--zoom-end",    default="2014-12-31")
    return p.parse_args()


def main():
    args      = parse_args()
    input_dir = Path(args.input_dir)
    out_dir   = Path(args.out_dir)

    spei_path = input_dir / f"{args.model}_{args.ssp}" / f"spei_{args.model}_{args.ssp}.nc"
    thr_path  = out_dir   / f"threshold_p{int(args.percentile*100):02d}_monthly_{args.model}_{args.ssp}.nc"
    cat_path  = out_dir   / f"drought_event_catalog_{args.model}_{args.ssp}.nc"

    for path, label in [(spei_path, "SPEI"), (thr_path, "Threshold"), (cat_path, "Catalog")]:
        if not path.exists():
            raise FileNotFoundError(f"{label} file not found: {path}")

    fig_root  = out_dir / "figures" / f"{args.model}_{args.ssp}"
    locations = parse_locations(args.locations)
    zoom      = (args.zoom_start, args.zoom_end)

    print(f"\n=== Diagnostics: {args.model} / {args.ssp} ===")

    print("\n[1/3] Threshold diagnostics...")
    plot_threshold_diagnostics(thr_path, fig_root / "threshold", args.spell_ref)

    print("\n[2/3] Point-level time series...")
    plot_timeseries(spei_path, thr_path, fig_root / "timeseries",
                    args.spei_var, args.spell_ref, locations, zoom)

    print("\n[3/4] Heatmaps...")
    plot_heatmaps(cat_path, thr_path, fig_root / "heatmaps")

    print("\n[4/4] Catalog distributions...")
    plot_catalog_distributions(cat_path, fig_root / "catalog")

    print(f"\nAll figures saved under: {fig_root}")

if __name__ == "__main__":
    main()