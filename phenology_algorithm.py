# Import packages
import os
import re
import gc
import h5py
import math
import folium
import rasterio
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray as rxr
import geopandas as gpd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from rasterio.mask import mask as rio_mask
from rasterio.transform import from_bounds
from shapely.ops import unary_union, linemerge, polygonize

import matplotlib.cm as cm
from matplotlib import colors
from matplotlib import dates, colors
from matplotlib.colors import Normalize
from matplotlib.colors import ListedColormap

from shapely.geometry import box
from shapely.geometry import Point
from shapely.geometry import shape

import boto3
from botocore import UNSIGNED
from botocore.client import Config

import glob
from scipy import stats
from rasterio.plot import plotting_extent
from rasterio.transform import array_bounds
from rasterio.warp import reproject
from rasterio.enums import Resampling

from PHENO_testing import *

# Regression helper functions
def calculate_r(observed, predicted):
    """
    Calculate r (Pearson's Correlation) between two time series
    """
    r = np.corrcoef(observed, predicted)[0, 1]
    return r

def calculate_r2(observed, predicted):
    """
    Calculate R^2 (Coefficient of Determination) between two time series
    """
    ss_res = np.sum((observed - predicted)**2)
    ss_tot = np.sum((observed - np.mean(observed))**2)
    r2 = 1 - ss_res / ss_tot
    return r2

def calculate_rmse(observed, predicted):
    """
    Calculate RMSE (Root Mean Square Error) between two time series
    """
    rmse = np.sqrt(np.nanmean((observed - predicted)**2))
    return rmse

# ----------------------------------------------------------------------------------
# --------------------------------Processing scripts--------------------------------
# ----------------------------------------------------------------------------------
def compute_hls_vi_stats(veg_index, data_dir, outdir,
                         tile_id, start_year=None, end_year=None,
                         roi_shp=None, roi_name=None):
    vi_pattern = re.compile(
        rf'^{re.escape(tile_id)}_10day_median_(\d{{7}})\_{veg_index.upper()}.tif$',
        re.IGNORECASE
    )
    doy_pattern = re.compile(
        rf'^{re.escape(tile_id)}_DOY_10day_(\d{{7}})\.tif$',
        re.IGNORECASE
    )

    vi_files  = {}
    doy_files = {}
    vi_list = []
    doy_list = []

    for fname in os.listdir(data_dir):
        m = vi_pattern.match(fname)
        if m:
            vi_files[m.group(1)] = os.path.join(data_dir, fname)
            continue
        m = doy_pattern.match(fname)
        if m:
            doy_files[m.group(1)] = os.path.join(data_dir, fname)

    years = sorted(set(k[:4] for k in vi_files))
    years = [y for y in years if
             (start_year is None or int(y) >= start_year) and
             (end_year   is None or int(y) <= end_year)]

    if not years:
        print(f"No data found for year range {start_year}–{end_year}")
        return None, None #pd.DataFrame()

    print(f"Processing {len(years)} year(s): {years[0]}–{years[-1]}")
    
    roi = None
    if roi_shp is not None:
        if Path(roi_shp).suffix == ".zip":
            roi = gpd.read_file(f"zip://{roi_shp}")
        else:
            roi = gpd.read_file(roi_shp)
        print(f"ROI loaded: {roi_shp}  ({len(roi)} feature(s))")
        if "LineString" in roi.geom_type.values:
            print("Shapefile contains LineStrings, merging into a Polygon.")
            # Create a polygon from the LineStrings
            boundary = linemerge(unary_union(roi.geometry))
            polys = list(polygonize(boundary))
            
            # Redefine the roi using the polygon
            roi = gpd.GeoDataFrame(
                geometry=polys,
                crs=roi.crs
            )

    records = []
    
    for year in years:
        year_keys       = sorted(k for k in vi_files if k[:4] == year)
        print(f"year_keys, {year_keys}")
        annual_vi_stack = []

        for doy_key in year_keys:
            vi_path  = vi_files[doy_key]
            doy_path = doy_files.get(doy_key)
            year_int   = int(doy_key[:4])
            doy_start  = int(doy_key[4:])
            start_date = datetime(year_int, 1, 1) + timedelta(days=doy_start - 1)

            vi = rxr.open_rasterio(vi_path, masked=True, chunks="auto").squeeze("band", drop=True)
            vi.attrs.pop("scale_factor", None)
            vi.attrs.pop("add_offset",   None)

            roi_reproj = None
            if roi is not None:
                roi_reproj = roi.to_crs(vi.rio.crs)
                vi = vi.rio.clip(roi_reproj.geometry, roi_reproj.crs, drop=True)

            valid = vi.values[~np.isnan(vi.values)]
            if valid.size == 0:
                print(f"Warning: No valid VI pixels for {doy_key} — skipping")
                # Store the crs in case the vi deleted is the last one before the annual
                vi_crs = vi.rio.crs
                del vi
                gc.collect()
                continue
            
            if doy_path:
                doy_da = rxr.open_rasterio(doy_path, masked=True).squeeze("band", drop=True)
                doy_da.attrs.pop("scale_factor", None)
                doy_da.attrs.pop("add_offset",   None)

                if roi_reproj is not None:
                    doy_da = doy_da.rio.clip(
                        roi_reproj.geometry,
                        roi_reproj.crs,
                        drop=True
                    )
                
                doy_da = doy_da.where(vi.notnull())
                doy_da = doy_da.where(doy_da <= 9)

                mean_doy_offset = float(doy_da.mean(skipna=True).values)
                if np.isnan(mean_doy_offset):
                    print(f"Warning: No valid DOY pixels for {doy_key}, using start date")
                    mean_doy_offset = 0.0
                rep_date = start_date + timedelta(days=round(mean_doy_offset))
                # Make sure the date doesn't cross into the next year for graphing purposes
                if rep_date.year != year_int:
                    rep_date = start_date

                doy_da = doy_da + doy_start
                
                try:
                    vi_list.append(vi.expand_dims(time=[start_date]))
                    doy_list.append(doy_da.expand_dims(time=[start_date]))
                except UnboundLocalError:
                    pass
            else:
                print(f"Warning: No DOY tif for {doy_key}, using start date")
                mean_doy_offset = 0.0
                rep_date        = start_date

            try:
                vi_list.append(vi.expand_dims(time=[start_date]))
                doy_list.append(doy_da.expand_dims(time=[start_date]))
            except UnboundLocalError:
                pass
            
            records.append({
                "doy_key":             int(doy_key),
                "composite_start":     start_date,
                "composite_mean_date": rep_date,
                "mean_doy_offset":     round(mean_doy_offset, 2),
                f"{veg_index}_mean":   float(np.mean(valid)),
                f"{veg_index}_median": float(np.median(valid)),
                f"{veg_index}_min":    float(np.min(valid)),
                f"{veg_index}_max":    float(np.max(valid)),
                f"{veg_index}_stdev":  float(np.std(valid, ddof=1)),
                "n_valid_pixels":      int(valid.size),
            })

            annual_vi_stack.append(vi.assign_coords(time=rep_date).expand_dims("time"))

        if not annual_vi_stack:
            print(f"No valid composites for {year} — skipping annual tif")
            continue

        stacked     = xr.concat(annual_vi_stack, dim="time")
        annual_mean = stacked.mean(dim="time", skipna=True)
        annual_std  = stacked.std( dim="time", skipna=True, ddof=1)

        annual_2band = xr.concat(
            [annual_mean.expand_dims("band"),
             annual_std.expand_dims("band")],
            dim="band"
        ).assign_coords(band=[1, 2])

        annual_2band.attrs = {
            "long_name":    [f"{veg_index.upper()}_annual_mean", f"{veg_index.upper()}_annual_stdev"],
            "year":         year,
            "n_composites": len(annual_vi_stack),
        }

        vi_crs = annual_vi_stack[0].rio.crs
        annual_2band = annual_2band.rio.write_crs(vi_crs)

        annual_fname = f"HLS_{tile_id}_{veg_index.upper()}_annual_{year}_{veg_index.upper()}.tif"
        annual_path  = os.path.join(outdir, annual_fname)
        annual_2band.rio.to_raster(annual_path, compress="lzw")
        print(f"Annual written: {annual_fname}  |  n_composites={len(annual_vi_stack)}")

        del stacked, annual_mean, annual_std, annual_2band, annual_vi_stack
        try:
            del vi
        except UnboundLocalError:
            pass
        gc.collect()
    
    try:
        df = pd.DataFrame(records).sort_values("composite_start").reset_index(drop=True)
        if roi_name:
            csv_path = os.path.join(outdir, f"HLS_{tile_id}_{veg_index.upper()}_timeseries_stats-{roi_name}.csv")
        else:
            csv_path = os.path.join(outdir, f"HLS_{tile_id}_{veg_index.upper()}_timeseries_stats.csv")
        df.to_csv(csv_path, index=False)
        print(f"Timeseries CSV: {csv_path}")
        
        vi_cube = xr.concat(vi_list, dim="time")
        doy_cube = xr.concat(doy_list, dim="time")
    
    except Exception as e:
        print(f"Could not create dataframe and/or cubes due to {e}.")
        df = pd.DataFrame()
        vi_cube = doy_cube = np.array([])
    
    return df, vi_cube, doy_cube

def compute_modis_vi_stats(veg_index, data_dir, outdir,
                           tile_id, start_year=None, end_year=None,
                           hls_data_dir=None, roi_shp=None, roi_name=None):

    vi_pattern = re.compile(
        rf'^MOD13Q1_{re.escape(veg_index.upper())}_(\d{{8}})\.tif$',
        re.IGNORECASE
    )
    doy_pattern = re.compile(
        r'^MOD13Q1_DOY_(\d{8})\.tif$',
        re.IGNORECASE
    )

    vi_files  = {}
    doy_files = {}
    vi_list = []
    doy_list = []

    for fname in os.listdir(data_dir):
        m = vi_pattern.match(fname)
        if m:
            vi_files[m.group(1)] = os.path.join(data_dir, fname)
            continue
        m = doy_pattern.match(fname)
        if m:
            doy_files[m.group(1)] = os.path.join(data_dir, fname)

    # Filter by year range on the YYYYMMDD key
    vi_files = {k: v for k, v in vi_files.items() if
                (start_year is None or int(k[:4]) >= start_year) and
                (end_year   is None or int(k[:4]) <= end_year)}

    if not vi_files:
        print(f"No data found for year range {start_year}–{end_year}")
        return pd.DataFrame()

    years = sorted(set(k[:4] for k in vi_files))
    years = [y for y in years if
             (start_year is None or int(y) >= start_year) and
             (end_year   is None or int(y) <= end_year)]

    print(f"Processing {len(years)} year(s): {years[0]}–{years[-1]}")

    # Build HLS bounding box to mask to MGRS tile (MODIS has larger footprint)
    hls_bbox_gdf = None
    if hls_data_dir is not None:
        hls_tifs = [
            os.path.join(hls_data_dir, f)
            for f in os.listdir(hls_data_dir)
            if f.lower().endswith(".tif")
        ]
        if hls_tifs:
            sample = rxr.open_rasterio(hls_tifs[0], masked=True)
            hls_crs    = sample.rio.crs
            hls_bounds = sample.rio.bounds()   # (left, bottom, right, top)
            del sample
            gc.collect()

            hls_bbox_geom = box(*hls_bounds)             
            hls_bbox_gdf  = gpd.GeoDataFrame(
                geometry=[hls_bbox_geom], crs=hls_crs
            )
            print(
                f"HLS tile bounds loaded from: {os.path.basename(hls_tifs[0])}\n"
                f"  CRS   : {hls_crs}\n"
                f"  Bounds: left={hls_bounds[0]:.2f}, bottom={hls_bounds[1]:.2f}, "
                f"right={hls_bounds[2]:.2f}, top={hls_bounds[3]:.2f}"
            )
        else:
            print("Warning: hls_data_dir provided but no .tif files found — skipping HLS bounds mask")

    roi = None
    if roi_shp is not None:
        if Path(roi_shp).suffix == ".zip":
            roi = gpd.read_file(f"zip://{roi_shp}")
        else:
            roi = gpd.read_file(roi_shp)
        print(f"ROI loaded: {roi_shp}  ({len(roi)} feature(s))")
        if "LineString" in roi.geom_type.values:
            print("Shapefile contains LineStrings, merging into a Polygon.")
            # Create a polygon from the LineStrings
            boundary = linemerge(unary_union(roi.geometry))
            polys = list(polygonize(boundary))
            
            # Redefine the roi using the polygon
            roi = gpd.GeoDataFrame(
                geometry=polys,
                crs=roi.crs
            )

    records = []

    for year in years:
        print(year)
        annual_vi_stack = []
        year_keys = sorted(k for k in vi_files if k[:4] == year)

        for doy_key in year_keys:
            vi_path  = vi_files[doy_key]
            doy_path = doy_files.get(doy_key)
            start_date = datetime.strptime(doy_key, '%Y%m%d')

            vi = rxr.open_rasterio(vi_path, masked=True, chunks="auto")
            
            # ── clip 1: HLS MGRS tile bounding box ───────────────────────────
            bbox_reproj = None
            if hls_bbox_gdf is not None:
                bbox_reproj = hls_bbox_gdf.to_crs(vi.rio.crs)
                vi = vi.rio.clip(bbox_reproj.geometry, bbox_reproj.crs,
                                 drop=True, from_disk=True)

            
            vi = vi.squeeze("band", drop=True)
            vi.attrs.pop("scale_factor", None)
            vi.attrs.pop("add_offset",   None)

            # ── clip 2: optional ROI ───────────────────────────
            roi_reproj = None
            if roi is not None:
                roi_reproj = roi.to_crs(vi.rio.crs)
                vi = vi.rio.clip(roi_reproj.geometry, roi_reproj.crs, drop=True)

            valid = vi.values[~np.isnan(vi.values)]
            if valid.size == 0:
                print(f"Warning: No valid VI pixels for {doy_key} — skipping")
                # Store the crs in case the vi deleted is the last one before the annual
                vi_crs = vi.rio.crs
                del vi
                gc.collect()
                continue
            
            if doy_path:
                doy_da = rxr.open_rasterio(doy_path, masked=True).squeeze("band", drop=True)
                doy_da.attrs.pop("scale_factor", None)
                doy_da.attrs.pop("add_offset",   None)

                if roi_reproj is not None:
                    doy_da = doy_da.rio.clip(roi_reproj.geometry, roi_reproj.crs, drop=True)

                doy_da = doy_da.rio.reproject_match(vi)
                doy_da = doy_da.where(vi.notnull() & (doy_da >= 1) & (doy_da <= 366))
                mean_doy = float(doy_da.mean(skipna=True).values)

                if np.isnan(mean_doy):
                    print(f"Warning: No valid DOY pixels for {doy_key}, using composite date")
                    rep_date = start_date
                else:
                    rep_date = datetime(start_date.year, 1, 1) + timedelta(days=round(mean_doy) - 1)
                    # Make sure the date doesn't cross into the next year for graphing purposes
                    if rep_date.year != start_date.year:
                        rep_date = start_date

                try:
                    vi_list.append(vi.expand_dims(time=[start_date]))
                    doy_list.append(doy_da.expand_dims(time=[start_date]))
                except UnboundLocalError:
                    pass
            else:
                print(f"Warning: No DOY tif for {doy_key}, using composite date")
                mean_doy = None
                rep_date = start_date
            
            records.append({
                "doy_key":             int(doy_key),
                "composite_date":      start_date,
                "composite_mean_date": rep_date,
                "mean_doy":            round(mean_doy, 2) if mean_doy is not None else None,
                f"{veg_index}_mean":   float(np.mean(valid)),
                f"{veg_index}_median": float(np.median(valid)),
                f"{veg_index}_min":    float(np.min(valid)),
                f"{veg_index}_max":    float(np.max(valid)),
                f"{veg_index}_stdev":  float(np.std(valid, ddof=1)),
                "n_valid_pixels":      int(valid.size),
            })

            annual_vi_stack.append(vi.assign_coords(time=rep_date).expand_dims("time"))

        if not annual_vi_stack:
            print(f"No valid composites for {year} — skipping annual tif")
            continue

        stacked     = xr.concat(annual_vi_stack, dim="time")
        annual_mean = stacked.mean(dim="time", skipna=True)
        annual_std  = stacked.std( dim="time", skipna=True, ddof=1)

        annual_2band = xr.concat(
            [annual_mean.expand_dims("band"),
             annual_std.expand_dims("band")],
            dim="band"
        ).assign_coords(band=[1, 2])

        annual_2band.attrs = {
            "long_name":    [f"{veg_index.upper()}_annual_mean", f"{veg_index.upper()}_annual_stdev"],
            "year":         year,
            "n_composites": len(annual_vi_stack),
        }

        vi_crs = annual_vi_stack[0].rio.crs
        annual_2band = annual_2band.rio.write_crs(vi_crs)

        annual_fname = f"MOD13Q1_{tile_id}_{veg_index.upper()}_annual_{year}_{veg_index.upper()}.tif"
        annual_path  = os.path.join(outdir, annual_fname)
        annual_2band.rio.to_raster(annual_path, compress="lzw")
        print(f"Annual written: {annual_fname}  |  n_composites={len(annual_vi_stack)}")

        del stacked, annual_mean, annual_std, annual_2band, annual_vi_stack
        try:
            del vi
        except UnboundLocalError:
            pass
        gc.collect()
    
    try:
        df = pd.DataFrame(records).sort_values("composite_date").reset_index(drop=True)
        if roi_name:
            csv_path = os.path.join(outdir, f"MOD13Q1_{veg_index.upper()}_timeseries_stats-{roi_name}.csv")
        else:
            csv_path = os.path.join(outdir, f"MOD13Q1_{veg_index.upper()}_timeseries_stats.csv")
        df.to_csv(csv_path, index=False)
        print(f"Timeseries CSV: {csv_path}")
    
        vi_cube = xr.concat(vi_list, dim="time")
        doy_cube = xr.concat(doy_list, dim="time")
    
    except Exception as e:
        print(f"Could not create dataframe and/or cubes due to {e}.")
        df = pd.DataFrame()
        vi_cube = doy_cube = np.array([])
    
    return df, vi_cube, doy_cube

# ----------------------------------------------------------------------------------
# --------------------------------Timeseries scripts--------------------------------
# ----------------------------------------------------------------------------------
def plot_vi_timeseries(
    hls_df,
    modis_df,
    veg_index,
    outdir,
    tile_id="HLS",
    stat="mean",
    plot_std=True,
    figsize=(14, 5),
    vi_min=None,        # ← new: set to None to disable filtering
):
    vi_col   = f"{veg_index.lower()}_{stat}"
    std_col  = f"{veg_index.lower()}_stdev"
    vi_upper = veg_index.upper()

    fig, ax = plt.subplots(figsize=figsize)

    # ── HLS timeseries ────────────────────────────────────────────────────────
    if not hls_df.empty and vi_col in hls_df.columns:
        hls_plot_df = hls_df.copy()
        hls_plot_df["composite_mean_date"] = pd.to_datetime(hls_plot_df["composite_mean_date"])
        hls_plot_df = hls_plot_df.sort_values("composite_mean_date").reset_index(drop=True)

        # ── filter below vi_min ───────────────────────────────────────────────
        if vi_min is not None:
            n_before = len(hls_plot_df)
            hls_plot_df = hls_plot_df[hls_plot_df[vi_col] >= vi_min].reset_index(drop=True)
            print(f"HLS: removed {n_before - len(hls_plot_df)} points below {vi_min} ({len(hls_plot_df)} remaining)")

        hls_x = hls_plot_df["composite_mean_date"]
        hls_y = hls_plot_df[vi_col]

        ax.plot(hls_x, hls_y,
                color="steelblue", linewidth=1.5, marker="o", markersize=3,
                label=f"HLS {vi_upper} ({stat})")

        if plot_std and std_col in hls_plot_df.columns:
            ax.fill_between(hls_x,
                            hls_y - hls_plot_df[std_col],
                            hls_y + hls_plot_df[std_col],
                            color="steelblue", alpha=0.15, label="HLS ±1σ")
    else:
        print(f"Warning: HLS DataFrame is empty or missing '{vi_col}' — skipping HLS plot")
        print(f"  Available columns: {list(hls_df.columns)}")

    # ── MODIS timeseries ──────────────────────────────────────────────────────
    if not modis_df.empty and vi_col in modis_df.columns:
        modis_plot_df = modis_df.copy()
        modis_plot_df["composite_mean_date"] = pd.to_datetime(modis_plot_df["composite_mean_date"])
        modis_plot_df = modis_plot_df.sort_values("composite_mean_date").reset_index(drop=True)

        # ── filter below vi_min ───────────────────────────────────────────────
        if vi_min is not None:
            n_before = len(modis_plot_df)
            modis_plot_df = modis_plot_df[modis_plot_df[vi_col] >= vi_min].reset_index(drop=True)
            print(f"MODIS: removed {n_before - len(modis_plot_df)} points below {vi_min} ({len(modis_plot_df)} remaining)")

        mod_x = modis_plot_df["composite_mean_date"]
        mod_y = modis_plot_df[vi_col]

        ax.plot(mod_x, mod_y,
                color="darkorange", linewidth=1.5, marker="s", markersize=3,
                label=f"MODIS {vi_upper} ({stat})")

        if plot_std and std_col in modis_plot_df.columns:
            ax.fill_between(mod_x,
                            mod_y - modis_plot_df[std_col],
                            mod_y + modis_plot_df[std_col],
                            color="darkorange", alpha=0.15, label="MODIS ±1σ")
    else:
        print(f"Warning: MODIS DataFrame is empty or missing '{vi_col}' — skipping MODIS plot")
        print(f"  Available columns: {list(modis_df.columns)}")

    # ── Formatting ────────────────────────────────────────────────────────────
    ax.set_title(f"{tile_id}  |  {vi_upper} Timeseries — HLS vs MODIS", fontsize=13)
    ax.set_xlabel("Date")
    ax.set_ylabel(vi_upper)
    ax.legend(loc="best", fontsize=9)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
    # plt.ylim(0, 0.5)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    fig.tight_layout()

    png_path = os.path.join(outdir, f"{tile_id}_{vi_upper}_HLS_vs_MODIS_timeseries.png")
    fig.savefig(png_path, dpi=150)
    plt.show()
    print(f"Plot saved: {png_path}")
    return fig, ax

def plot_vi_seasonal_mean(
    hls_df,
    modis_df,
    veg_index,
    outdir,
    tile_id="HLS",
    stat="mean",
    vi_min=None,
    hls_bin_width=10,      # matches HLS 10-day composite cadence
    modis_bin_width=16,    # matches MODIS 16-day composite cadence
    figsize=(13, 5),
):
    """
    Seasonal plot showing the multi-year mean ± 1 std for each sensor
    on a single set of axes. Each sensor uses its native composite
    cadence as the bin width so DOY bins are not artificially widened.

    Parameters
    ----------
    hls_df          : DataFrame from compute_hls_vi_stats()
    modis_df        : DataFrame from compute_modis_vi_stats()
    veg_index       : str    e.g. 'NDVI'
    outdir          : str    directory to save PNG
    tile_id         : str    used in title / filename
    stat            : str    'mean' or 'median'
    vi_min          : float | None   drop composites below this value
    hls_bin_width   : int    DOY bin width for HLS   (default 10)
    modis_bin_width : int    DOY bin width for MODIS (default 16)
    figsize         : tuple
    """

    vi_col   = f"{veg_index.lower()}_{stat}"
    vi_upper = veg_index.upper()

    # ── helper: prep + bin ────────────────────────────────────────────────────
    def _prep(df, date_col, bin_width):
        d = df.copy()
        d[date_col] = pd.to_datetime(d[date_col])
        d = d.sort_values(date_col).reset_index(drop=True)
        d["_year"] = d[date_col].dt.year
        d["_doy"]  = d[date_col].dt.dayofyear
        if vi_min is not None:
            d = d[d[vi_col] >= vi_min].reset_index(drop=True)

        # Bin to sensor-native cadence
        d["_doy_bin"] = (((d["_doy"] - 1) // bin_width) * bin_width
                         + bin_width // 2 + 1)
        return d

    # ── helper: cross-year mean & std per DOY bin ─────────────────────────────
    def _seasonal_stats(df):
        return (
            df.groupby("_doy_bin")[vi_col]
            .agg(bin_mean="mean", bin_std="std", n_years="count")
            .reset_index()
            .sort_values("_doy_bin")
        )

    # ── per-sensor config — note separate bin_width per sensor ────────────────
    sensor_cfg = {
        "HLS":   dict(df=hls_df,   date_col="composite_mean_date",
                      color="steelblue",  marker="o",
                      bin_width=hls_bin_width),
        "MODIS": dict(df=modis_df, date_col="composite_mean_date",
                      color="darkorange", marker="s",
                      bin_width=modis_bin_width),
    }

    fig, ax = plt.subplots(figsize=figsize)

    for sensor, cfg in sensor_cfg.items():
        df_in = cfg["df"]
        if df_in.empty or vi_col not in df_in.columns:
            print(f"Warning: {sensor} DataFrame empty or missing '{vi_col}' — skipping")
            continue

        prepped = _prep(df_in, cfg["date_col"], cfg["bin_width"])
        stats   = _seasonal_stats(prepped)

        x     = stats["_doy_bin"].values
        ymean = stats["bin_mean"].values
        ystd  = stats["bin_std"].fillna(0).values    # single-year bins → std=NaN → 0

        # ── mean line ─────────────────────────────────────────────────────────
        ax.plot(x, ymean,
                color=cfg["color"], linewidth=2.0,
                marker=cfg["marker"], markersize=4,
                label=f"{sensor} {vi_upper} mean  (bin={cfg['bin_width']}d)",
                zorder=3)

        # ── ±1 std shading ────────────────────────────────────────────────────
        ax.fill_between(x,
                        ymean - ystd,
                        ymean + ystd,
                        color=cfg["color"], alpha=0.18,
                        label=f"{sensor} ±1σ",
                        zorder=2)

    # ── month x-tick labels ───────────────────────────────────────────────────
    month_doys   = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"]
    ax.set_xticks(month_doys)
    ax.set_xticklabels(month_labels)
    ax.set_xlim(1, 365)

    ax.set_title(f"{tile_id}  |  {vi_upper} Multi-Year Seasonal Mean ± 1σ  "
                 f"(HLS {hls_bin_width}d  |  MODIS {modis_bin_width}d)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel(vi_upper)
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)
    fig.tight_layout()

    png_path = os.path.join(outdir, f"{tile_id}_{vi_upper}_seasonal_mean_std.png")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Plot saved: {png_path}")
    return fig, ax

# ----------------------------------------------------------------------------------
# --------------------------------Phenology scripts---------------------------------
# ----------------------------------------------------------------------------------
import time
from scipy.integrate import trapezoid
import os
import tempfile
from joblib import Parallel, delayed

import warnings
warnings.filterwarnings('ignore', message='invalid value encountered in cast')

# from phenometrics_utils import *
from scipy.interpolate import LSQUnivariateSpline
import bottleneck

def _make_worker_slices(ny: int, n_workers: int) -> list[tuple[int, int]]:
    """ Takes n_rows of data and n_workers and calculates slice coords for each worker """
    base, extra = divmod(ny, n_workers)
    slices, start = [], 0
    for i in range(n_workers):
        end = start + base + (1 if i < extra else 0)
        if start < end:
            slices.append((start, end))
        start = end
    return slices


def _process_worker_slice(
    vi_mmap_path:       str,
    doy_mmap_path:      str,
    cube_shape:         tuple,          # (n_times, ny, nx)
    # row range owned by this worker
    row_start:          int,
    row_end:            int,
    t_nominal:          np.ndarray,     # (n_times,) shared time axis
    weights_template:   np.ndarray,     # (n_times,) Gaussian-decay weights
    t_daily:            np.ndarray,     # (n_output,) evaluation points
    min_valid_points:   int,
    value_min:          float,
    value_max:          float,
    fill_low_data:      str,
    k:                  int,
    n_output:           int,
    use_context_months: bool,
    target_year_offset: float,
    sensor:             str = "HLS",
) -> tuple[int, int, np.ndarray]:

    vi_data  = np.memmap(vi_mmap_path,  dtype=np.float32,
                          mode="r", shape=cube_shape)
    if doy_mmap_path is not None:
        doy_data = np.memmap(doy_mmap_path, dtype=np.float32,
                             mode="r", shape=cube_shape)
    
    n_rows = row_end - row_start
    nx     = cube_shape[2]
    result = np.full((n_output, n_rows, nx), np.nan, dtype=np.float32)
    # Store regression alongside result
    r_out    = np.full((n_rows, nx), np.nan, np.float32)
    r2_out   = np.full((n_rows, nx), np.nan, np.float32)
    rmse_out = np.full((n_rows, nx), np.nan, np.float32)

    # for each local row 
    for local_yi, yi in enumerate(range(row_start, row_end)):
        # process all x pixels in row yi
        for xi in range(nx):
            
            # 1. Extract single x,y pixel time series
            ts      = vi_data[:, yi, xi]
            if doy_mmap_path is not None:
                # t_pixel = doy_data[:, yi, xi]
                # Change doy based on ref_date
                doy = doy_data[:, yi, xi]
                t_pixel = target_year_offset + doy - 1
                valid   = np.isfinite(ts) & np.isfinite(t_pixel)
            else:
                t_pixel = t_nominal
                valid   = np.isfinite(ts)

            n_valid = valid.sum()

            # 2. Low-data handling (ToDo: add in fill logic) 
            if n_valid < min_valid_points:
                if fill_low_data == "mean" and n_valid > 0:
                    result[:, local_yi, xi] = np.nanmean(ts[valid])
                continue   # leave as NaN for default

            x_valid = t_pixel[valid]
            y_valid = ts[valid].astype(np.float64)
            w_valid = weights_template[valid].copy()

            # 3. Monotonicity check + deduplication
            if not np.all(np.diff(x_valid) > 1e-6):
                idx     = np.argsort(x_valid, kind="stable")
                x_valid = x_valid[idx]
                y_valid = y_valid[idx]
                w_valid = w_valid[idx]
                # Remove duplicate x positions (LSQ requires strictly increasing)
                keep    = np.concatenate([[True], np.diff(x_valid) > 1e-6])
                x_valid = x_valid[keep]
                y_valid = y_valid[keep]
                w_valid = w_valid[keep]

            if len(x_valid) < min_valid_points:
                continue

            # 4. Up-weight EVI extremes to allow spline to capture peaks/troughs
            y_range = y_valid.max() - y_valid.min()
            if y_range > 0.1 and use_context_months:
                lo = y_valid.min() + 0.20 * y_range
                hi = y_valid.min() + 0.80 * y_range
                w_valid[y_valid < lo] *= 2.0
                w_valid[y_valid > hi] *= 2.0

            # 5. Knots: trim the precomputed set to this pixel's range
            # x_range  = x_valid[-1] - x_valid[0]            
            if use_context_months:
                if sensor == "HLS":
                    n_knots = min(max(len(x_valid) // 3, 12), len(x_valid) - k - 1)
                # Use less knots for MODIS
                elif sensor == "MODIS":
                    n_knots = min(max(len(x_valid) // 5, 3), len(x_valid) - k - 2)
                    
                # Originally chose knots using a percentile but changed to indices to ensure
                # enough observations between knots
                # interior = np.unique(
                #     np.percentile(x_valid, np.linspace(10, 90, n_knots))
                # )
                # interior = interior[
                #     (interior > x_valid[0]) & (interior < x_valid[-1])
                # ]
                
                # Try indices to leave k+1 points at each end
                idx = np.linspace(k + 1, len(x_valid) - k - 2, n_knots, dtype=int)
                interior = x_valid[idx]
                interior = np.unique(interior)
                
                if len(interior) > 1:
                    keep     = np.concatenate([[True], np.diff(interior) >= 10.0])
                    interior = interior[keep]
                if len(interior) < 3:
                    continue

            # 6. Fit spline to full context observation dates and then evaluate on daily ts
            # if not np.all(np.diff(interior) > 0):
            #     print(interior)
            assert np.all(np.diff(x_valid) > 0)
            
            try:
                if not use_context_months:
                    peak_idx    = np.argmax(y_valid)
                    
                    # peak_t      = x_valid[peak_idx]
                    # pre_peak_t  = x_valid[0] + (peak_t - x_valid[0]) * 0.5
                    # post_peak_t = peak_t     + (x_valid[-1] - peak_t) * 0.5
                    # interior    = np.array([pre_peak_t, peak_t, post_peak_t])
                    # interior    = interior[
                    #     (interior > x_valid[0] + 1) & (interior < x_valid[-1] - 1)
                    # ]
                    
                    # Use indices to make sure there are observations between the knots
                    pre_idx = max(1, peak_idx // 2)
                    post_idx = peak_idx + (len(x_valid) - peak_idx) // 2
                    post_idx = min(len(x_valid) - 2, post_idx)
                    peak_idx = np.clip(peak_idx, 1, len(x_valid) - 2)
                    interior = np.unique(
                        x_valid[[pre_idx, peak_idx, post_idx]]
                    )
                    
                    if len(interior) < 2:
                        continue
                    spl = LSQUnivariateSpline(x_valid, y_valid, interior, w=w_valid, k=2) # k=3
                    y_pred = spl(x_valid)
                    r_out[local_yi, xi] = calculate_r(y_valid, y_pred)
                    r2_out[local_yi, xi] = calculate_r2(y_valid, y_pred)
                    rmse_out[local_yi, xi] = calculate_rmse(y_valid, y_pred)
                else:
                    spl = LSQUnivariateSpline(x_valid, y_valid, interior, w=w_valid, k=k)
                    y_pred = spl(x_valid)
                    r_out[local_yi, xi] = calculate_r(y_valid, y_pred)
                    r2_out[local_yi, xi] = calculate_r2(y_valid, y_pred)
                    rmse_out[local_yi, xi] = calculate_rmse(y_valid, y_pred)
            
                result[:, local_yi, xi] = np.clip(
                    spl(t_daily), value_min, value_max
                ).astype(np.float32)
                
            except Exception as e:
                print(f"\nPixel ({yi}, {xi}) failed:")
                print(f"n_obs      = {len(x_valid)}")
                if use_context_months:
                    print(f"k          = {k}")
                else:
                    print("k          = 3")
                print(f"n_knots    = {len(interior)}")
                print(f"x range    = {x_valid[0]} -> {x_valid[-1]}")
                print(f"x diff min = {np.min(np.diff(x_valid))}")
                
                print("x_valid:", x_valid)
                print("knots:", interior)
                
                print(e)
    
                if fill_low_data == "mean":
                    result[:, local_yi, xi] = float(np.nanmean(y_valid))

    return row_start, row_end, result, r_out, r2_out, rmse_out

def _process_worker_slice_savgol(
    vi_mmap_path:     str,
    doy_mmap_path:    str,
    cube_shape:       tuple,          # (n_times, ny, nx)
    row_start:        int,
    row_end:          int,
    t_nominal:        np.ndarray,
    t_daily:          np.ndarray,
    min_valid_points: int,
    value_min:        float,
    value_max:        float,
    fill_low_data:    str,
    n_output:         int,
    window_length:    int,
    polyorder:        int,
) -> tuple[int, int, np.ndarray]:
    """
    Savitzky-Golay smoothing worker.
    Linearly interpolates sparse obs to daily then applies SG filter.
    """
    from scipy.signal import savgol_filter

    vi_data = np.memmap(vi_mmap_path, dtype=np.float32, mode="r", shape=cube_shape)
    if doy_mmap_path is not None:
        doy_data = np.memmap(doy_mmap_path, dtype=np.float32, mode="r", shape=cube_shape)
    n_rows   = row_end - row_start
    nx       = cube_shape[2]
    result   = np.full((n_output, n_rows, nx), np.nan, dtype=np.float32)

    for local_yi, yi in enumerate(range(row_start, row_end)):
        for xi in range(nx):
            ts = vi_data[:, yi, xi]
            if doy_mmap_path is not None:
                # Change doy based on ref_date
                t_pixel = doy_data[:, yi, xi]
                valid = np.isfinite(ts) & np.isfinite(t_pixel)
            else:
                t_pixel = t_nominal
                valid = np.isfinite(ts)

            if valid.sum() < min_valid_points:
                if fill_low_data == "mean" and valid.sum() > 0:
                    result[:, local_yi, xi] = np.nanmean(ts[valid])
                continue

            x_valid = t_pixel[valid]
            y_valid = ts[valid].astype(np.float64)

            # 1. Linear interpolation to daily grid
            daily_interp = np.interp(t_daily, x_valid, y_valid)

            # 2. SG filter — ensure window_length doesn't exceed series length
            wl = min(window_length, len(daily_interp))
            wl = wl if wl % 2 == 1 else wl - 1   # must be odd
            if wl <= polyorder:
                result[:, local_yi, xi] = daily_interp.astype(np.float32)
                continue

            try:
                smoothed = savgol_filter(daily_interp, window_length=wl, polyorder=polyorder)
                result[:, local_yi, xi] = np.clip(
                    smoothed, value_min, value_max
                ).astype(np.float32)
            except Exception:
                if fill_low_data == "mean":
                    result[:, local_yi, xi] = float(np.nanmean(y_valid))

    return row_start, row_end, result

def smooth_vi_chunk_for_year(
    chunk:                xr.DataArray,
    target_year:          int,
    doy_data:             xr.DataArray = None,
    sensor:               str = "HLS",
    # --- algorithm config ---
    smoother:             str   = "spline", # spline | savgol
    savgol_window:        int   = 31,   # days odd
    savgol_polyorder:     int   = 3,       
    min_valid_points:     int   = 8,
    min_valid_frac:       float = 0.30,
    fill_low_data:        str   = "nan",  # currently no gap filling, Bolton uses the context years to "grab" similar values but that's a weaker method 
    value_min:            float = -1.0,
    value_max:            float = 1.0,
    daily_output:         bool  = True,
    k:                    int   = 5,      # spline degree, 4 = cubic
    use_context_months:   bool  = True,   # computed to avoid pits/peaks from overfitting gaps
    testing_mode:         bool  = False,
    _pool:                Parallel | None = None,  # warm pool from caller
    n_jobs=-1,
) -> xr.DataArray:
    """
    Fit a pixel-wise LSQ smoothing spline over a ±context_months window
    around target_year, returning daily smoothed VI for target_year only.

    Parameters
    ----------
    chunk                : (time, y, x) VI DataArray covering at least
                           target_year ± context_months of data.
    target_year          : Year to produce output for.
    min_valid_points     : Pixels with fewer finite observations are skipped.
    fill_low_data        : "nan" — leave skipped pixels as NaN.
                           "mean" — fill with the pixel's temporal mean.
    value_min/max        : Output clipping bounds.
    k                    : Spline degree (5 recommended for EVI phenology).
    doy_data             : Optional (time, y, x) actual-DOY DataArray for
                           10-day composites. When provided with
                           composite_start_doys, each pixel gets its own
                           time axis derived from actual observation DOYs.
    composite_start_doys : 1-D array of composite-period start DOYs aligned
                           to chunk.time. Required when doy_data is provided.
    testing_mode         : If True, output spans the full fitting window
                           instead of target_year only (used for QC plots).
    _pool                : Pre-warmed joblib.Parallel instance. Pass this in
                           from process_all_chunks_yearly so the loky pool
                           startup cost is paid once per run, not per chunk.

    Returns
    -------
    xr.DataArray : (time, y, x) daily smoothed VI.
                   time = 365 days of target_year (or full context window if testing_mode).
    """
    t_start   = time.time()
    if _pool is not None and hasattr(_pool, 'n_jobs'):
        pool_workers = _pool.n_jobs
        n_workers    = os.cpu_count() if pool_workers == -1 else max(1, pool_workers)
        print(f"  Workers   : {n_workers} (from warm pool)")
    else:
        n_workers = os.cpu_count() if n_jobs == -1 else max(1, n_jobs)
        print(f"  Workers   : {n_workers} (local pool)")

    if use_context_months:
        context_months = 12
    else:
        context_months = 0
        
    # ----------------------------------------------------------------
    # 1. Temporal subset — restrict to context window (should be this window incoming)
    # ----------------------------------------------------------------
    fit_start = (pd.Timestamp(f"{target_year}-01-01")
                 - pd.DateOffset(months=context_months))
    fit_end   = (pd.Timestamp(f"{target_year}-12-31")
                 + pd.DateOffset(months=context_months))

    fit_chunk = chunk.sel(time=slice(fit_start, fit_end))
    if doy_data is not None:
        fit_doy = doy_data.sel(time=slice(fit_start, fit_end))
    else:
        fit_doy = None
    
    if fit_chunk.sizes["time"] == 0:
        raise ValueError(
            f"No data in fitting window {fit_start.date()} – {fit_end.date()}. "
            f"Check that target_year={target_year} is within the loaded data range."
        )

    # ----------------------------------------------------------------
    # 2. Drop entirely-NaN timesteps
    # ----------------------------------------------------------------
    valid_ts  = np.any(np.isfinite(fit_chunk.values), axis=(1, 2))
    n_dropped = int((~valid_ts).sum())
    if n_dropped > 0:
        fit_chunk = fit_chunk.isel(time=valid_ts)

    # dim size of data (n timesteps, y pixel cnt, x pixel cnt)
    n_times, ny, nx = fit_chunk.shape
    n_pixels        = ny * nx

    # ----------------------------------------------------------------
    # 3. Calc effective min_valid_points
    #     Hard floor  : k+1 (minimum for a degree-k spline)
    #     Hard ceiling: n_times (can't require more points than time steps exist)
    # ----------------------------------------------------------------        
    K_FLOOR     = k + 1                                    # e.g. 6 for k=5
    frac_floor  = int(np.ceil(n_times * min_valid_frac))   # e.g. 11 from 35×0.3

    effective_min_valid = min(
        max(K_FLOOR, frac_floor),    # adaptive floor when large amount of data
        min_valid_points,            # args ceiling — lowers threshold if < floor
        n_times,                     # hard ceiling - don't overfit
    )

    print(f"  min_valid : {effective_min_valid} "
          f"(k+1={K_FLOOR}, "
          f"{min_valid_frac*100:.0f}%×{n_times}={frac_floor}, "
          f"user={min_valid_points}) "
          f"  effective={effective_min_valid}")

    print(f"  Chunk     : {ny}×{nx} = {n_pixels:,} pixels | "
          f"{n_times} timesteps ({n_dropped} all-NaN dropped) | "
          f"min_valid={effective_min_valid}")
        
    # ----------------------------------------------------------------
    # 4. Nominal time axis
    #    Days since (target_year-1)-01-01 — keeps values in a sensible
    #    range for spline fit regardless of year in context window
    # ----------------------------------------------------------------
    ref_date  = np.datetime64(f"{target_year - 1}-01-01")
    t_nominal = ((fit_chunk.time.values - ref_date)
                 / np.timedelta64(1, "D")).astype(np.float64)
    target_year_offset = float(
        (np.datetime64(f"{target_year}-01-01") - ref_date)
        / np.timedelta64(1, "D")
    )

    if len(t_nominal) == 0:
        print(f"   WARNING: 0 valid timesteps for {target_year} current chunk;"
              f"     chunk is entirely masked. Returning NaN output.")
        daily_times = pd.date_range(f"{target_year}-01-01", f"{target_year}-12-31", freq="D")        
        nan_data = np.full(
            (len(daily_times), fit_chunk.shape[1], fit_chunk.shape[2]),
            np.nan,
            dtype=np.float32,
        )
        return xr.DataArray(
            nan_data,
            dims=["time", "y", "x"],
            coords={"time": daily_times, "y": fit_chunk.y, "x": fit_chunk.x,},
        )
        
    if len(t_nominal) < min_valid_points:
        print(f"  [smooth_vi] WARNING: only {len(t_nominal)} valid timesteps ")
        daily_times = pd.date_range(f"{target_year}-01-01", f"{target_year}-12-31", freq="D")
        nan_data = np.full(
            (len(daily_times), fit_chunk.shape[1], fit_chunk.shape[2]),
            np.nan,
            dtype=np.float32,
        )
        return xr.DataArray(
            nan_data,
            dims=["time", "y", "x"],
            coords={
                "time": daily_times,
                "y":    fit_chunk.y,
                "x":    fit_chunk.x,
            },
        )
        
    # ----------------------------------------------------------------
    # 5. Output time axis - infill days so EVI2 is continuous across DOY of target-year
    # ----------------------------------------------------------------
    daily_dates = (pd.date_range(fit_start, fit_end)
                   if testing_mode
                   else pd.date_range(f"{target_year}-01-01",
                                      f"{target_year}-12-31"))
    t_daily  = ((daily_dates.values - ref_date)
                / np.timedelta64(1, "D")).astype(np.float64)
    n_output = len(daily_dates)
    t_daily = np.clip(t_daily, t_nominal[0], t_nominal[-1])
    print(f"t_nominal: {t_nominal[0]:.1f} - {t_nominal[-1]:.1f}", flush=True)
    print(f"t_daily:   {t_daily[0]:.1f}  - {t_daily[-1]:.1f}", flush=True)
    print(f"overlap:   {t_daily[0] >= t_nominal[0]} to {t_daily[-1] <= t_nominal[-1]}", flush=True)
    
    # ----------------------------------------------------------------
    # 6. Gaussian-decay weight template
    #    Observations near the centre of target_year get full weight;
    #    context observations are downweighted by distance
    # ----------------------------------------------------------------
    target_center = float(
        (np.datetime64(f"{target_year}-07-01") - ref_date)
        / np.timedelta64(1, "D")
    )
    days_from_center = np.abs(t_nominal - target_center)
    weights_template = np.exp(-0.25 * days_from_center / 365) * 0.85 + 0.15
    
    # ----------------------------------------------------------------
    # 8. Write memmap temp files for distributed processing
    #    Parent writes once - workers read zero-copy via OS page mapping.
    #    TemporaryDirectory cleans up automatically on exit.
    # ----------------------------------------------------------------
    cube_shape    = (n_times, ny, nx)
    smoothed_out = np.full((n_output, ny, nx), np.nan, dtype=np.float32)
    # Store regression
    r_all    = np.full((ny, nx), np.nan, np.float32)
    r2_all   = np.full((ny, nx), np.nan, np.float32)
    rmse_all = np.full((ny, nx), np.nan, np.float32)

    with tempfile.TemporaryDirectory(prefix="smooth_vi_") as tmpdir:

        # VI memmap
        vi_path = str(Path(tmpdir) / "vi.mmap")
        vi_mm       = np.memmap(vi_path, dtype=np.float32, mode="w+", shape=cube_shape)
        vi_mm[:]    = fit_chunk.values.astype(np.float32)
        vi_mm.flush(); del vi_mm
        
        # DOY memmap
        if fit_doy is not None:
            doy_path = str(Path(tmpdir) / "doy.mmap")
            doy_mm = np.memmap(doy_path, dtype=np.float32, mode="w+", shape=cube_shape)
            doy_mm[:] = fit_doy.values.astype(np.float32)
            doy_mm.flush(); del doy_mm
        else:
            doy_path = None

        # ----------------------------------------------------------------
        # 10. Dispatch
        #     One task per worker, each covering ~ny/n_workers rows of data.
        #     Use a warm pool if provided, otherwise create a local one.
        # ----------------------------------------------------------------
        worker_slices = _make_worker_slices(ny, n_workers)
        assert len(worker_slices) == n_workers, (
            f"Slice count {len(worker_slices)} != worker count {n_workers} — "
            f"check n_jobs/pool alignment"
        )
        print(f"  Knot mode : {'sparse/Arctic' if not use_context_months else 'full context'} | "  f"k={k}")
        print(f"  Workers   : {n_workers} processes | Output: {n_output} days")
        print(f"  Slices    : {len(worker_slices)} × ~{ny // n_workers} rows each")
        t_dispatch = time.time()

        # Shared kwargs — same for every worker
        if smoother == "savgol":
            worker_kwargs = dict(
                vi_mmap_path     = vi_path,
                doy_mmap_path    = doy_path,
                cube_shape       = cube_shape,
                t_nominal        = t_nominal,
                t_daily          = t_daily,
                min_valid_points = effective_min_valid,
                value_min        = value_min,
                value_max        = value_max,
                fill_low_data    = fill_low_data,
                n_output         = n_output,
                window_length    = savgol_window,
                polyorder        = savgol_polyorder,
            )
            worker_fn = _process_worker_slice_savgol
        else:
            worker_kwargs = dict(
                vi_mmap_path       = vi_path,
                doy_mmap_path      = doy_path,
                cube_shape         = cube_shape,
                t_nominal          = t_nominal,
                weights_template   = weights_template,
                t_daily            = t_daily,
                min_valid_points   = effective_min_valid,
                value_min          = value_min,
                value_max          = value_max,
                fill_low_data      = fill_low_data,
                k                  = k,
                n_output           = n_output,
                use_context_months = use_context_months,
                target_year_offset = target_year_offset,
                sensor             = sensor,
            )
            worker_fn = _process_worker_slice
        print(f"  Smoother  : {smoother}"
              + (f" (window={savgol_window}, polyorder={savgol_polyorder})"
                if smoother == "savgol" else f" (k={k})"))
        
        executor = _pool or Parallel(
            n_jobs=n_workers, prefer="processes", batch_size="auto"
        )

        # use precomputed row-wise worker slices to distribute with kwargs to workers
        results = executor(
            delayed(worker_fn)(row_start=s, row_end=e, **worker_kwargs)
            for s, e in worker_slices
        )

        # ----------------------------------------------------------------
        # 11. Reassemble
        # ----------------------------------------------------------------
        n_fitted = n_skipped = 0
        if smoother == "savgol":
            for row_start, row_end, row_result, in results:
                smoothed_out[:, row_start:row_end, :] = row_result
                finite_mask = np.any(np.isfinite(row_result), axis=0)   # (n_rows, nx)
                n_fitted  += int(finite_mask.sum())
                n_skipped += int((~finite_mask).sum())
        else:
            for row_start, row_end, row_result, row_r, row_r2, row_rmse in results:
                smoothed_out[:, row_start:row_end, :] = row_result
                finite_mask = np.any(np.isfinite(row_result), axis=0)   # (n_rows, nx)
                n_fitted  += int(finite_mask.sum())
                n_skipped += int((~finite_mask).sum())
                # Reassemble regression metrics
                r_all[row_start:row_end] = row_r
                r2_all[row_start:row_end] = row_r2
                rmse_all[row_start:row_end] = row_rmse
            # Print regression summary
            print("Spline regression summary")
            print("r:")
            print(f"mean = {np.nanmean(r_all):.3f}")
            print(f"min  = {np.nanmin(r_all):.3f}")
            print(f"max  = {np.nanmax(r_all):.3f}")
            print("R2:")
            print(f"mean = {np.nanmean(r2_all):.3f}")
            print(f"min  = {np.nanmin(r2_all):.3f}")
            print(f"max  = {np.nanmax(r2_all):.3f}")
            print("RMSE:")
            print(f"mean = {np.nanmean(rmse_all):.3f}")
            print(f"min  = {np.nanmin(rmse_all):.3f}")
            print(f"max  = {np.nanmax(rmse_all):.3f}")

    # ----------------------------------------------------------------
    # 12. Timing summary
    # ----------------------------------------------------------------
    t_total    = time.time() - t_start
    t_compute  = time.time() - t_dispatch
    rate       = n_pixels / max(t_compute, 1e-6)
    print(f"  Done      : {n_fitted:,} fitted | {n_skipped:,} skipped | "
          f"{t_total:.1f}s total | {rate:,.0f} px/s")

    gc.collect()

    # ----------------------------------------------------------------
    # 13. Return as xr.DataArray
    # ----------------------------------------------------------------
    return xr.DataArray(
        smoothed_out,
        dims=["time", "y", "x"],
        coords={
            "time": daily_dates,
            "y":    fit_chunk.y,
            "x":    fit_chunk.x,
        },
    )


def apply_thresholds_chunk(chunk: xr.DataArray,
                           min_val: float = 0.1,
                           max_val: float = 0.95) -> xr.DataArray:
    """Apply min/max thresholds to chunk."""
    return chunk.where((chunk >= min_val) & (chunk <= max_val))


# per Bolton et al., 2020 eq.3 pg4
def despike_timeseries_chunk(
        chunk: xr.DataArray,
        max_gap_days: int = 45,
        abs_threshold: float = 0.1,
        rel_threshold: float = 2.0,
        handle_edges: bool = True,
) -> xr.DataArray:
    """
    Three-point de-spiking with optional per-pixel DOY awareness.

    Args:
        chunk:                DataArray (time, y, x) of VI values
        doy_data:             DataArray (time, y, x) of DOY offsets within composite
        composite_start_doys: Array of composite start DOY per timestep
        max_gap_days:         Max gap between pre/post for despiking
        abs_threshold:        Absolute difference threshold
        rel_threshold:        Relative difference threshold
        handle_edges:         Check first/last observations for spikes
    """
    n_times = len(chunk.time)
    chunk_values = chunk.values  # (time, y, x)

    times = pd.to_datetime(chunk.time.values)
    if len(times) == 0:
        return chunk        
    nominal_days = (times - times[0]).days.astype(np.float32)
    spike_mask = np.zeros_like(chunk_values, dtype=bool)
    time_days_da = xr.DataArray(nominal_days, dims=['time'],
                                coords={'time': chunk.time})
    
    vi_pre   = chunk.ffill(dim="time").shift(time=1)    
    vi_post  = chunk.bfill(dim="time").shift(time=-1)  
    time_pre  = time_days_da.ffill(dim="time").shift(time=1)
    time_post = time_days_da.bfill(dim="time").shift(time=-1)

    gap = time_post - time_pre
    weight = (time_days_da - time_pre) / (time_post - time_pre)
    vi_fit = vi_pre + (vi_post - vi_pre) * weight

    amplitude = vi_post - vi_pre
    diff = vi_fit - chunk
    abs_diff = np.abs(diff)
    rel_diff = np.abs(diff / amplitude.where(np.abs(amplitude) > 0.001))

    spike_da = (
            (abs_diff > abs_threshold)
            & (rel_diff > rel_threshold)
            & (gap < max_gap_days)
            & (~vi_pre.isnull())
            & (~vi_post.isnull())
    )
    spike_mask = spike_da.values

    if handle_edges and n_times >= 3:
        # ── First obs ─────────────────────────────────────────────
        t_gap_first = nominal_days[1] - nominal_days[0]
        if t_gap_first < max_gap_days:
            diff_first = np.abs(chunk_values[0] - chunk_values[1])
            spike_mask[0] = (
                (diff_first > abs_threshold * 1.5)
                & (~np.isnan(chunk_values[0]))
                & (~np.isnan(chunk_values[1]))
            )
    
        # ── Second obs — must be low relative to BOTH neighbours ──
        t_gap_second = nominal_days[2] - nominal_days[0]
        if t_gap_second < max_gap_days:
            spike_mask[1] = (
                (chunk_values[1] < chunk_values[0])   # lower than first
                & (chunk_values[1] < chunk_values[2]) # lower than third
                & ((np.abs(chunk_values[1] - chunk_values[0]) > abs_threshold) | (np.abs(chunk_values[1] - chunk_values[2]) > abs_threshold))
                & (~np.isnan(chunk_values[0]))
                & (~np.isnan(chunk_values[1]))
                & (~np.isnan(chunk_values[2]))
            )
    
        # ── Last obs ──────────────────────────────────────────────
        t_gap_last = nominal_days[-1] - nominal_days[-2]
        if t_gap_last < max_gap_days:
            diff_last = np.abs(chunk_values[-1] - chunk_values[-2])
            spike_mask[-1] = (
                (diff_last > abs_threshold * 1.5)
                & (~np.isnan(chunk_values[-1]))
                & (~np.isnan(chunk_values[-2]))
            )
    
    chunk_despiked = chunk.where(~spike_mask)
    
    n_spikes = int(spike_mask.sum())
    n_total = int((~chunk.isnull()).sum())
    if n_spikes > 0:
        pct = 100 * n_spikes / n_total if n_total > 0 else 0
        print(f"  De-spiking: removed {n_spikes} spikes ({pct:.2f}%) [nominal gaps]")
    
    return chunk_despiked


def compute_scene_quality_metrics(
    chunk: xr.DataArray,
    target_year: int,
) -> tuple[np.ndarray, np.ndarray]:

    chunk_target_year = chunk.where(chunk.time.dt.year == target_year)
    values = chunk_target_year.values                                   
    doys   = chunk_target_year.time.dt.dayofyear.values.astype(np.float32) 

    valid_mask  = ~np.isnan(values)                        
    valid_count = valid_mask.sum(axis=0).astype(np.float32) 

    # Mask invalid timesteps and compute per-pixel DOY range
    doys_3d    = np.broadcast_to(doys[:, None, None], values.shape)
    doys_valid = np.where(valid_mask, doys_3d, np.nan) 

    doy_max = np.nanmax(doys_valid, axis=0)
    doy_min = np.nanmin(doys_valid, axis=0)

    with np.errstate(invalid='ignore', divide='ignore'):
        mean_revisit = np.select(
            condlist=[
                valid_count > 1,    # normal case
                valid_count == 1,   # single observation → fill with 1
            ],
            choicelist=[
                (doy_max - doy_min) / (valid_count - 1),
                np.ones_like(valid_count),
            ],
            default=np.nan,         # 0 valid observations
        ).astype(np.float32)

    quality_pixels = np.where(
        valid_count > 0, valid_count, np.nan
    ).astype(np.float32)

    return mean_revisit, quality_pixels

    
def annual_phenometrics_chunk(chunk: xr.DataArray,
                              year: int = None,
                              threshold_greenup_pct: float = 0.15) -> dict[str, np.ndarray]:
    """
    Calculate annual phenometrics for a chunk.

    Args:
        chunk: DataArray (time, y, x) - should span multiple years
        doy_data: Optional DataArray (time, y, x) of actual observation DOY
                  (for composites where DOY varies per pixel)
        year: Specific year to process (None = all years in data)
        threshold_greenup_pct: Percentage of amplitude for greenup/dormancy thresholds (default 15%)
        composite_start_doys: Array of start DOY for each time step (for 10day composites)

    Returns:
        Dict with 3D arrays (year, y, x) for each metric
    """

    ny, nx = chunk.shape[1], chunk.shape[2]
    n_years = 1  # n years

    # Initialize output phenometric arrays
    annual_mean = np.full((n_years, ny, nx), np.nan, dtype=np.float32)
    annual_max = np.full((n_years, ny, nx), np.nan, dtype=np.float32)
    annual_max_doy = np.full((n_years, ny, nx), np.nan, dtype=np.float32)
    annual_min = np.full((n_years, ny, nx), np.nan, dtype=np.float32)
    annual_min_doy = np.full((n_years, ny, nx), np.nan, dtype=np.float32)

    greenup_vi = np.full((n_years, ny, nx), np.nan, dtype=np.float32)
    greenup_doy = np.full((n_years, ny, nx), np.nan, dtype=np.float32)
    greenup_threshold = np.full((n_years, ny, nx), np.nan, dtype=np.float32)

    dormancy_vi = np.full((n_years, ny, nx), np.nan, dtype=np.float32)
    dormancy_doy = np.full((n_years, ny, nx), np.nan, dtype=np.float32)

    annual_amplitude = np.full((n_years, ny, nx), np.nan, dtype=np.float32)
    growing_season_length = np.full((n_years, ny, nx), np.nan, dtype=np.float32)

    auc_full = np.full((n_years, ny, nx), np.nan, dtype=np.float32)
    auc_net = np.full((n_years, ny, nx), np.nan, dtype=np.float32)

    greenup_rate = np.full((n_years, ny, nx), np.nan, dtype=np.float32)
    greenup_rate_doy = np.full((n_years, ny, nx), np.nan, dtype=np.float32)

    senescence_rate = np.full((n_years, ny, nx), np.nan, dtype=np.float32)
    senescence_rate_doy = np.full((n_years, ny, nx), np.nan, dtype=np.float32)
    
    year_vi = chunk
    if len(year_vi.time) == 0:
        return None

    nominal_doys = year_vi.time.dt.dayofyear.values

    # 1. MEAN
    annual_mean[0] = year_vi.mean(dim='time').values

    # 2. MAX
    year_max = year_vi.max(dim='time').values
    annual_max[0] = year_max

    # 3. MIN
    year_min = year_vi.min(dim='time').values
    annual_min[0] = year_min

    # 4. AMPLITUDE
    amplitude = year_max - year_min
    annual_amplitude[0] = amplitude

    # 5. Max DOY
    fill_value = -9999
    year_filled = year_vi.fillna(fill_value)
    max_indices = year_filled.argmax(dim='time').values  # (y, x)

    # 6. Min DOY
    min_fill_value = 9999
    year_min_filled = year_vi.fillna(min_fill_value)
    min_indices = year_min_filled.argmin(dim='time').values

    # prep annual values and mask
    year_vi_values = year_vi.values  # (time, y, x)
    all_nan_mask = year_vi.isnull().all(dim='time').values  # (y, x)

    for yi in range(ny):
        for xi in range(nx):
            if not all_nan_mask[yi, xi]:
                max_idx = max_indices[yi, xi]
                annual_max_doy[0, yi, xi] = nominal_doys[max_idx]
                min_idx = min_indices[yi, xi]
                annual_min_doy[0, yi, xi] = nominal_doys[min_idx]

    # 7. GREENUP, 8. DORMANCY, 9. AUC full, 10. AUC net, 11. GREENUP Inflection, 12. Senescences Inflection
    threshold = year_min + (amplitude * threshold_greenup_pct)
    greenup_threshold[0] = threshold
    for yi in range(ny):
        for xi in range(nx):
            if all_nan_mask[yi, xi]:
                continue

            pixel_vi = year_vi_values[:, yi, xi]
            pixel_max_doy = annual_max_doy[0, yi, xi] 
            pixel_threshold = threshold[yi, xi]

            if np.isnan(pixel_max_doy) or np.isnan(pixel_threshold):
                continue

            pixel_doys = nominal_doys.astype(float)

            valid_pixel = ~np.isnan(pixel_vi) & ~np.isnan(pixel_doys)

            if valid_pixel.sum() < 3:
                continue

            pixel_vi_valid = pixel_vi[valid_pixel]
            pixel_doys_valid = pixel_doys[valid_pixel]
            
            # Greenup: first DOY exceeding threshold BEFORE peak AND on ascending segment
            pre_peak_mask = pixel_doys < pixel_max_doy
            if pre_peak_mask.sum() > 0:
                pre_peak_vi = pixel_vi[pre_peak_mask]
                pre_peak_doys = pixel_doys[pre_peak_mask]
                valid = ~np.isnan(pre_peak_vi)
                
                if valid.sum() > 2:
                    pre_peak_vi  = pre_peak_vi[valid]
                    pre_peak_doys = pre_peak_doys[valid]

                    # Compute derivative on pre-peak valid obs
                    pre_deriv = np.gradient(pre_peak_vi, pre_peak_doys)
                    above_thresh = pre_peak_vi >= pixel_threshold
                    ascending    = pre_deriv > 0
                    valid_greenup = above_thresh & ascending

                    if valid_greenup.any():
                        first_idx = np.argmax(valid_greenup)
                        greenup_doy[0, yi, xi] = pre_peak_doys[first_idx]
                        greenup_vi[0, yi, xi] = pre_peak_vi[first_idx]

            # Dormancy: first DOY below threshold AFTER peak
            post_peak_mask = pixel_doys > pixel_max_doy            
            if post_peak_mask.sum() > 0:
                vi_post = pixel_vi[post_peak_mask]
                doy_post = pixel_doys[post_peak_mask]         

                # reduce to non NAN range of dates
                valid_post = ~np.isnan(vi_post)
                vi_post   = vi_post[valid_post]
                doy_post   = doy_post[valid_post]
                if len(vi_post) == 0:
                    # No valid post-peak obs: dormancy = NaN
                    continue
                
                crossing_idx = np.where(vi_post <= pixel_threshold)[0]            
                if len(crossing_idx) > 0:
                    i = crossing_idx[0]            
                    if i > 0:
                        x0, x1 = doy_post[i-1], doy_post[i]
                        y0, y1 = vi_post[i-1], vi_post[i]            
                        dormancy_doy[0, yi, xi] = (doy_post[i-1] 
                                                   if abs(y0 - pixel_threshold) < abs(y1 - pixel_threshold) 
                                                   else doy_post[i])
                        dormancy_vi[0, yi, xi] = pixel_threshold
                    else:
                        dormancy_doy[0, yi, xi] = doy_post[i]
                        dormancy_vi[0, yi, xi] = vi_post[i]
            
                else:
                    # fallback: last valid observation after peak
                    dormancy_doy[0, yi, xi] = doy_post[-1]
                    dormancy_vi[0, yi, xi] = vi_post[-1]
            
            # 11. AUC FULL and 12. AUC NET
            pix_greenup = greenup_doy[0, yi, xi]
            pix_dormancy = dormancy_doy[0, yi, xi]
            pix_min = year_min[yi, xi]

            if not np.isnan(pix_greenup) and not np.isnan(pix_dormancy):
                gs_mask = (pixel_doys_valid >= pix_greenup) & (pixel_doys_valid <= pix_dormancy)

                if gs_mask.sum() >= 3:
                    gs_doy = pixel_doys_valid[gs_mask]
                    gs_vi = pixel_vi_valid[gs_mask]

                    gs_valid = ~np.isnan(gs_vi)
                    if gs_valid.sum() >= 3:
                        gs_doy = gs_doy[gs_valid]
                        gs_vi = gs_vi[gs_valid]

                        # AUC Full: total area under curve from greenup to dormancy
                        auc_full[0, yi, xi] = trapezoid(gs_vi, gs_doy)

                        # AUC Net: area above the minimum baseline
                        gs_vi_above_min = gs_vi - pix_min
                        auc_net[0, yi, xi] = trapezoid(gs_vi_above_min, gs_doy)

            # 13 & 14. INFLECTION POINTS (steepest greenup and senescence)
            if len(pixel_doys_valid) >= 4:
                vi_derivative = np.gradient(pixel_vi_valid, pixel_doys_valid)
                min_rise_scalar = 0.10
                # Steepest greenup: max positive derivative before peak
                pre_peak = pixel_doys_valid < pixel_max_doy
                if pre_peak.sum() >= 2:
                    pre_derivs = vi_derivative[pre_peak]
                    pre_doys   = pixel_doys_valid[pre_peak]
                    pre_vi    = pixel_vi_valid[pre_peak]
                
                    # ── Exclude peak shoulder ─────────────────────────────────
                    # Greenup inflection must be after VI has risen meaningfully
                    # from baseline — at least 20% of amplitude above min
                    pixel_min_vi  = annual_min[0, yi, xi]
                    amplitude_px   = annual_amplitude[0, yi, xi]
                    min_rise       = pixel_min_vi + (amplitude_px * min_rise_scalar)
                
                    on_ascending_limb = pre_vi >= min_rise
                    if on_ascending_limb.sum() >= 2:
                        pre_derivs = pre_derivs[on_ascending_limb]
                        pre_doys   = pre_doys[on_ascending_limb]
                
                        max_rate_idx = np.argmax(pre_derivs)
                        greenup_rate[0, yi, xi]     = pre_derivs[max_rate_idx]
                        greenup_rate_doy[0, yi, xi] = pre_doys[max_rate_idx]

                # Steepest senescence: max negative derivative after peak
                post_peak = pixel_doys_valid > pixel_max_doy
                if post_peak.sum() >= 2:
                    post_derivs = vi_derivative[post_peak]
                    post_doys = pixel_doys_valid[post_peak]
                    post_vi    = pixel_vi_valid[post_peak]
                    
                    pixel_peak_vi = annual_max[0, yi, xi]
                    amplitude_px   = annual_amplitude[0, yi, xi]
                    min_drop       = pixel_peak_vi - (amplitude_px * min_rise_scalar)

                    on_descending_limb = post_vi <= min_drop
                    if on_descending_limb.sum() >= 2:
                        post_derivs = post_derivs[on_descending_limb]
                        post_doys   = post_doys[on_descending_limb]
                
                        min_rate_idx = np.argmin(post_derivs)
                        senescence_rate[0, yi, xi]     = post_derivs[min_rate_idx]
                        senescence_rate_doy[0, yi, xi] = post_doys[min_rate_idx]
                
    # 15. Growing season length
    valid_both = ~np.isnan(greenup_doy[0]) & ~np.isnan(dormancy_doy[0])
    growing_season_length[0, valid_both] = dormancy_doy[0, valid_both] - greenup_doy[0, valid_both]

    return {
        'annual_mean': annual_mean,
        'annual_max': annual_max,
        'annual_min': annual_min,
        'annual_max_doy': annual_max_doy,
        'annual_amplitude': annual_amplitude,
        'greenup_doy': greenup_doy,
        'dormancy_doy': dormancy_doy,
        'growing_season_length': growing_season_length,

        'annual_min_doy': annual_min_doy,
        'greenup_vi': greenup_vi,
        'dormancy_vi': dormancy_vi,
        'greenup_threshold': greenup_threshold,

        'auc_full': auc_full,
        'auc_net': auc_net,
        'greenup_rate': greenup_rate,
        'greenup_rate_doy': greenup_rate_doy,
        'senescence_rate': senescence_rate,
        'senescence_rate_doy': senescence_rate_doy,
    }


def get_context_months_from_gaps(
    chunk: xr.DataArray,
    target_year: int,
    gap_threshold_days: int = 45,
) -> bool:
    """
    Check the actual observation record.
    If the first observation gap at start or end of target year
    exceeds gap_threshold_days, context years will cause edge spikes.
    Return 0 context months in that case.
    """
    # # Check if chunk has the context
    # min_date = pd.Timestamp(chunk.time.min().values)
    # max_date = pd.Timestamp(chunk.time.max().values)
    
    # # Chunk should have data from previous and next years
    # required_start = pd.Timestamp(f"{target_year}-01-01") - pd.DateOffset(months=12)
    # required_end   = pd.Timestamp(f"{target_year}-12-31") + pd.DateOffset(months=12)

    # if min_date > required_start or max_date < required_end:
    #     return False

    
    target_obs = chunk.sel(time=str(target_year))
    has_obs = target_obs.notnull().any(dim=["y", "x"])
    if not has_obs.any():
        return False   # all-NaN chunk — handled elsewhere
    
    obs_times = target_obs.time.values
    valid_times = obs_times[has_obs.values]

    # Gap from Jan 1 to first observation
    jan1 = np.datetime64(f"{target_year}-01-01")
    dec31 = np.datetime64(f"{target_year}-12-31")
    gap_start = int((valid_times[0]  - jan1)  / np.timedelta64(1, 'D'))
    gap_end   = int((dec31 - valid_times[-1]) / np.timedelta64(1, 'D'))

    if gap_start >= gap_threshold_days or gap_end >= gap_threshold_days:
        return False

    return True

def calc_obs_snow_background(
    chunk:                    xr.DataArray,
    threshold_background_pct: float = 0.15,  # fraction of amplitude above min
    low_pct:                  float = 0.10,  # percentile of all valid obs
    snow_doy_start:           int   = 300,
    snow_doy_end:             int   = 100,
    min_snow_obs:             int   = 3,
) -> xr.DataArray:

    doy       = chunk.time.dt.dayofyear
    snow_mask = (doy >= snow_doy_start) | (doy <= snow_doy_end)
    snow_obs  = chunk.isel(time=snow_mask)
    n_valid   = snow_obs.notnull().sum(dim="time")

    # Path 1: winter obs available: low percentile of winter window
    snow_background = (
        snow_obs
        .quantile(low_pct, dim="time", skipna=True)
        .drop_vars("quantile", errors="ignore")
        .clip(min=0.0)
    )

    # Path 2: no winter obs (Arctic)
    #   low percentile of ALL valid obs + amplitude fraction
    #   this sits at the dormant floor, not dragged by noise/edge obs
    all_low = (
        chunk
        .quantile(low_pct, dim="time", skipna=True)
        .drop_vars("quantile", errors="ignore")
        .clip(min=0.0)
    )
    chunk_min = chunk.min(dim="time", skipna=True)
    chunk_max = chunk.max(dim="time", skipna=True)
    amplitude = chunk_max - chunk_min

    # floor = low percentile + small amplitude fraction
    # prevents background from sitting below real dormant signal
    amplitude_background = (all_low + amplitude * threshold_background_pct).clip(min=0.0)

    print(f"  snow_background    : min={float(snow_background.min()):.4f} "
          f"mean={float(snow_background.mean()):.4f} "
          f"max={float(snow_background.max()):.4f}")
    print(f"  amplitude_background: min={float(amplitude_background.min()):.4f} "
          f"mean={float(amplitude_background.mean()):.4f} "
          f"max={float(amplitude_background.max()):.4f}")

    background = xr.where(n_valid >= min_snow_obs, snow_background, amplitude_background)
    background = background.drop_vars("quantile", errors="ignore")

    print(f"  final background   : min={float(background.min()):.4f} "
          f"mean={float(background.mean()):.4f} "
          f"max={float(background.max()):.4f}")

    return background
    

def full_pipeline_chunk(chunk: xr.DataArray,
                        doy_data: xr.DataArray = None,
                        apply_threshold: bool = True,
                        min_vi_threshold: float = -1.0,
                        max_vi_threshold: float = 1.0,
                        threshold_greenup_pct: float = 0.15,
                        fill_snow_gaps: bool = False,
                        despike: bool = True,
                        despike_max_gap: int = 45,
                        despike_abs_threshold: float = 0.1,
                        despike_rel_threshold: float = 2.0,
                        use_infill: bool = True,
                        gap_threshold_days: int = 45,
                        smoother: str = "spline",
                        target_year: int = None,
                        testing_mode: bool = False,
                        _pool = None,
                        n_jobs:int = -1,
                        sensor: str = "HLS",
                        **kwargs) -> dict[str, np.ndarray]:
    """
    Full processing pipeline for a chunk.

    Pipeline:
        1. Apply VI thresholds: ensures any anomalous VI values are clipped
        2. TBD Positive/bright pixel filtering
        3. De-spike (three-point method): removes
        4. Interpolate gaps
        5. Calculate annual phenometrics

    """

    if sensor == "MODIS":
        # MODIS is smoother, so it has smaller spikes
        despike_abs_threshold = 0.05
        # MODIS has 16-day composites vs HLS 10-day, so it can have larger gaps
        gap_threshold_days = 60

    metric_mapping = {
        'annual_mean': 'mean_vi',
        'annual_max': 'max_vi',
        'annual_min': 'min_vi',
        'annual_max_doy': 'max_doy',
        'annual_amplitude': 'amplitude',
        'greenup_doy': 'greenup_doy',
        'dormancy_doy': 'dormancy_doy',
        'growing_season_length': 'growing_season_length',
        'annual_min_doy': 'min_doy',
        'greenup_vi': 'greenup_vi',
        'dormancy_vi': 'dormancy_vi',
        'greenup_threshold': 'greenup_threshold',
        'auc_full': 'auc_full',
        'auc_net': 'auc_net',
        'greenup_rate': 'greenup_rate',
        'greenup_rate_doy': 'greenup_rate_doy',
        'senescence_rate': 'senescence_rate',
        'senescence_rate_doy': 'senescence_rate_doy'
        # 'mean_revisit_time': 'mean_revisit_time',
        # 'quality_pixel_cnt': 'quality_pixel_cnt'
    }
    
    chunk_original = chunk.copy(deep=True) if testing_mode else None

    # Step 1: Threshold
    print("Step 1: Thresholding")
    if apply_threshold:
        chunk = apply_thresholds_chunk(
            chunk,
            min_vi_threshold,
            max_vi_threshold
        )

    chunk_post_threshold = chunk.copy(deep=True) if testing_mode else None

    # TODO Step: Positive/Bright pixel filtering (blue and red bands)
    
    # Step 2: Negative pixel filtering using DOY (EVI2 despiking - cloud shadows)
    # - uses target year +/- 1 year, if edge case remove the non-existing year
    if despike:
        print("Step 2: Despiking")
        chunk = despike_timeseries_chunk(
            chunk,
            max_gap_days=despike_max_gap,
            abs_threshold=despike_abs_threshold,
            rel_threshold=despike_rel_threshold,
        )
        target_obs_despiked = chunk.sel(time=str(target_year))
        target_obs_raw  = chunk_post_threshold.sel(time=str(target_year)) if testing_mode else None
        if testing_mode:
            removed = target_obs_raw.notnull() & target_obs_despiked.isnull()
            removed_times = target_obs_raw.time[removed.any(dim=["y", "x"])].values
            
            print(f"  Despiked dates in {target_year}:")
            for t in removed_times:
                print(f"    {pd.Timestamp(t).date()}")
            
    chunk_post_despike = chunk.copy(deep=True) #if testing_mode else None

    # -----------------------------------------------------------------------------------------
    # Step 2b: Context-year gap infill
    # Uses despiked observations from context years to fill gaps in the
    # target year before the spline sees the data.
    # context_infill_diagnostics = None
    # target_da_for_spline = chunk.sel(time=str(target_year))  # default: no infill

    # use_context_months = get_context_months_from_gaps(chunk=chunk, target_year=target_year)
    # context_years_present = sorted({
    #     int(y) for y in chunk.time.dt.year.values
    #     if int(y) != target_year
    # })

    # if len(context_years_present) >= 1 and use_infill == True:
    #     print("Step 2b: Context-year observation infill")
    #     target_da_for_spline, context_infill_diagnostics = build_context_infilled_observations(
    #         chunk_despiked  = chunk,             # full 3-yr despiked DataArray
    #         target_year     = target_year,
    #         n_harmonics     = 3,
    #         min_similarity  = 0.60,              # tune: lower = more permissive infill
    #         scale_to_target = True,
    #         testing_mode    = testing_mode,
    #     )
    #     # Rebuild a chunk that contains the infilled target year so the spline
    #     # fitter receives the augmented observations
    #     other_years = chunk.sel(
    #         time=~chunk.time.dt.year.isin([target_year])
    #     )
    #     chunk_for_spline = xr.concat(
    #         [other_years, target_da_for_spline],
    #         dim="time"
    #     ).sortby("time")
    # else:
    #     print("Step 2b: Context infill skipped "
    #             f"(use_context_months={use_context_months}, "
    #             f"context_years={context_years_present})")
    #     chunk_for_spline = chunk

    # chunk_post_context_infill = chunk_for_spline.copy(deep=True) if testing_mode else None

    # Step 3: calculate scene revisit and quality pixels before the spline fit, 365 DOY data is generated
    print("Step 3: Scene quality metrics")
    scene_mean_revisit, scene_quality_pixels = compute_scene_quality_metrics(chunk, target_year)

    valid_timesteps = (~np.isnan(chunk.values)).any(axis=(1, 2)).sum()
    if valid_timesteps == 0:
        print(f"  WARNING: chunk has 0 valid timesteps for {target_year}. All metrics will be NaN — skipping spline and phenometrics.")
        return {
            f'{name}_{target_year}': np.full((chunk.shape[1], chunk.shape[2]), np.nan, dtype=np.float32)
                for name in metric_mapping.values()
        } | {
            f'mean_revisit_time_{target_year}': scene_mean_revisit,
            f'quality_pixel_cnt_{target_year}': scene_quality_pixels,
        }
        
    # Step 4: apply penalized cubic spline interpolation
    print("Step 4: Apply interpolation")
    use_context_months = get_context_months_from_gaps(chunk=chunk,target_year=target_year,gap_threshold_days=gap_threshold_days)    
    fill_snow_gaps     = not use_context_months 
    print(f"  use_context_months : {use_context_months}")
    print(f"  fill_snow_gaps     : {fill_snow_gaps}")  
    
    smoothed_daily = smooth_vi_chunk_for_year(
        chunk,
        doy_data=doy_data,
        target_year=target_year,
        smoother = smoother,
        testing_mode=testing_mode,
        use_context_months=use_context_months,
        _pool=_pool,
        n_jobs=n_jobs,
        sensor=sensor,
    )
    
    chunk_post_spline = smoothed_daily.copy(deep=True) #if testing_mode else None

    # Testing fill_snow_gaps
    # fill_snow_gaps = True
    
    if fill_snow_gaps:
        # Step 5: Fill snow gaps using naive min EVI2 value
        print("Step 5: Snow gap fill", flush=True)
        background_threshold = calc_obs_snow_background(chunk) 
        target_obs  = chunk.sel(time=str(target_year))
        is_valid = target_obs.notnull()
        has_any   = is_valid.any(dim="time")              
        first_idx = is_valid.argmax(dim="time").compute()            
        last_idx  = (target_obs.sizes["time"] - 1 
                     - is_valid.isel(time=slice(None, None, -1)).argmax(dim="time")).compute()                                           
        first_obs_doy = target_obs.time.dt.dayofyear.isel(time=first_idx).where(has_any)
        last_obs_doy  = target_obs.time.dt.dayofyear.isel(time=last_idx).where(has_any)
        smoothed_year = smoothed_daily.sel(time=str(target_year))
        daily_doy     = smoothed_year.time.dt.dayofyear
        bg            = background_threshold.drop_vars("quantile", errors="ignore")
    
        before_first = daily_doy < first_obs_doy
        after_last   = daily_doy > last_obs_doy
        
    #     no_data      = ~has_any
    
    #     # Spline value at the exact boundary day [y, x]
    #     first_doy_idx  = (daily_doy == first_obs_doy)
    #     last_doy_idx   = (daily_doy == last_obs_doy)
    
    #     spline_at_first = smoothed_year.where(first_doy_idx).max(dim="time")
    #     spline_at_last  = smoothed_year.where(last_doy_idx).max(dim="time")
    
    #     # Fill = min(background, spline at boundary) — never step up OR down
    #     lead_fill  = xr.where(spline_at_first < bg, spline_at_first, bg)
    #     trail_fill = xr.where(spline_at_last  < bg, spline_at_last,  bg)
    
    #     smoothed_year = smoothed_year.where(~(before_first | no_data), other=lead_fill)
    #     smoothed_year = smoothed_year.where(~(after_last   | no_data), other=trail_fill)    
    #     smoothed_year_pheno = smoothed_daily.sel(time=str(target_year)).where(
    #         (daily_doy >= first_obs_doy) & (daily_doy <= last_obs_doy)
    #     )
    #     chunk_post_snow_fill = smoothed_year.copy(deep=True) if testing_mode else None
    # else:
    #     smoothed_year_pheno = smoothed_daily.sel(time=str(target_year))
    #     chunk_post_snow_fill = None
        outside_obs  = (before_first | after_last | ~has_any)

        # Spline is kept where it sits above the floor; clamped to floor where below
        smoothed_year = xr.where(
            outside_obs,
            xr.ufuncs.maximum(smoothed_year, bg),   # let spline continue if above bg
            smoothed_year                            # inside obs window: untouched
        )
    
        # ── Transition DOYs ───────────────────────────────────────────────
        spline_above_bg = smoothed_year > bg
        rising_idx      = spline_above_bg.argmax(dim="time")
        falling_idx     = (spline_above_bg.sizes["time"] - 1
                           - spline_above_bg.isel(time=slice(None, None, -1))
                                            .argmax(dim="time"))
        any_above       = spline_above_bg.any(dim="time")
        rising_doy      = daily_doy.isel(time=rising_idx ).where(any_above)
        falling_doy     = daily_doy.isel(time=falling_idx).where(any_above)

        # ── Prevent post-season spline rebound ────────────────────────────
        vals     = smoothed_year.values.copy()   # (T, ny, nx)
        doys_1d  = daily_doy.values              # (T,)
        bg_vals  = bg.values                     # (ny, nx)
        ny, nx   = vals.shape[1], vals.shape[2]

        rise_doy  = rising_doy.values            # (ny, nx)
        fall_doy  = falling_doy.values           # (ny, nx)

        # Spline value at the exact transition DOYs → ceiling for outside region
        rise_idx_1d = np.argmin(np.abs(doys_1d[:, None, None] - rise_doy[None]), axis=0)
        fall_idx_1d = np.argmin(np.abs(doys_1d[:, None, None] - fall_doy[None]), axis=0)

        # (ny, nx) — the spline value at each pixel's transition DOY
        spline_at_rise = vals[rise_idx_1d, np.arange(ny)[:, None], np.arange(nx)[None, :]]
        spline_at_fall = vals[fall_idx_1d, np.arange(ny)[:, None], np.arange(nx)[None, :]]

        for t_idx in range(len(doys_1d)):
            d           = doys_1d[t_idx]
            before_rise = d < rise_doy                  # (ny, nx)
            after_fall  = d > fall_doy

            # Pre-season: clamp to [bg, spline_at_rise]
            vals[t_idx] = np.where(
                before_rise,
                np.clip(vals[t_idx], bg_vals, spline_at_rise),
                vals[t_idx]
            )
            # Post-season: clamp to [bg, spline_at_fall]
            vals[t_idx] = np.where(
                after_fall,
                np.clip(vals[t_idx], bg_vals, spline_at_fall),
                vals[t_idx]
            )

        smoothed_year = smoothed_year.copy(data=vals)

        smoothed_year_pheno = smoothed_year.where(
            (daily_doy >= rising_doy) & (daily_doy <= falling_doy)
        )

    else:
        smoothed_year = smoothed_daily
        smoothed_year_pheno = smoothed_daily.sel(time=str(target_year))
        
    chunk_post_snow_fill = smoothed_year.copy(deep=True) if testing_mode else None

    # background_threshold = calc_obs_snow_background(chunk) 
    target_obs  = chunk_post_despike.sel(time=str(target_year))
    is_valid = target_obs.notnull()
    has_any   = is_valid.any(dim="time")              
    first_idx = is_valid.argmax(dim="time").compute()            
    last_idx  = (target_obs.sizes["time"] - 1 - is_valid.isel(time=slice(None, None, -1)).argmax(dim="time")).compute()                                           
    first_obs_doy = target_obs.time.dt.dayofyear.isel(time=first_idx).where(has_any)
    last_obs_doy  = target_obs.time.dt.dayofyear.isel(time=last_idx).where(has_any)
    # bg            = background_threshold.drop_vars("quantile", errors="ignore")
    # first_obs = chunk_post_despike.sel(time=str(target_year)).isel(time=first_idx).where(has_any)
    # last_obs = chunk_post_despike.sel(time=str(target_year)).isel(time=last_idx).where(has_any)

    daily_doy = smoothed_year_pheno.time.dt.dayofyear
    spline_year = smoothed_daily.sel(time=str(target_year))
    spline_at_first = (
        spline_year
        .where(daily_doy == first_obs_doy)
        .max(dim="time")
    )
    spline_at_last = (
        spline_year
        .where(daily_doy == last_obs_doy)
        .max(dim="time")
    )
    
    if testing_mode:
        target_chunk_post_snow_fill = chunk_post_snow_fill.sel(time=str(target_year))
        daily_doy     = target_chunk_post_snow_fill.time.dt.dayofyear
        before_first = daily_doy < first_obs_doy
        after_last   = daily_doy > last_obs_doy
        # outside_obs  = (before_first | after_last)
        
        # Spline is kept within the range where there are observations
        target_chunk_post_snow_fill = xr.where(
            before_first,
            spline_at_first,
            xr.where(
                after_last,
                spline_at_last,
                target_chunk_post_snow_fill
            )
        )
        chunk_post_snow_fill.loc[{"time": target_chunk_post_snow_fill.time}] = target_chunk_post_snow_fill

    daily_doy = smoothed_year_pheno.time.dt.dayofyear
    before_first = daily_doy < first_obs_doy
    after_last   = daily_doy > last_obs_doy
    # outside_obs  = (before_first | after_last)
    
    smoothed_year_pheno = xr.where(
        before_first,
        spline_at_first,
        xr.where(
            after_last,
            spline_at_last,
            smoothed_year_pheno
        )
    )
    
    # Step 6: Annual phenometrics
    # smoothed_year = smoothed_daily.where(smoothed_daily.time.dt.year == target_year)        
    print("Step 6: Calculate phenometrics")
    pheno = annual_phenometrics_chunk(
        smoothed_year_pheno,
        threshold_greenup_pct=threshold_greenup_pct,
        year=target_year,
    )
    results = {}
    for internal_name, output_name in metric_mapping.items():
        results[f'{output_name}_{target_year}'] = pheno[internal_name][0]
        
    results[f'mean_revisit_time_{target_year}'] = scene_mean_revisit
    results[f'quality_pixel_cnt_{target_year}'] = scene_quality_pixels

    if testing_mode:
        results['_intermediate'] = {
            'original': chunk_original,
            'post_threshold': chunk_post_threshold,
            'post_despike': chunk_post_despike,
            'post_spline': chunk_post_spline,
            'post_snow_fill': chunk_post_snow_fill,
        }

    return results

def phenometrics(vi_cube, doy_cube, sensor, smoother, veg_index, outdir, tile_id, start_year, end_year):
    # Grab the bounds and crs from the vi_cube (the doy_cube should have the same values)
    bounds = vi_cube.rio.bounds()
    crs = vi_cube.rio.crs
    
    for year in range(start_year, end_year+1):
        window_start = pd.Timestamp(f"{year-1}-01-01")
        window_end   = pd.Timestamp(f"{year+1}-12-31")

        # Implement a 3-year window around target_year instead of loading the
        # entire cubes
        vi_window = vi_cube.sel(
            time=slice(window_start, window_end)
        )
        doy_window = doy_cube.sel(
            time=slice(window_start, window_end)
        )
        results = full_pipeline_chunk(
            vi_window,
            doy_data=doy_window,
            target_year=year,
            sensor=sensor,
            smoother=smoother,
        )
        
        for metric in results:
            # Get height and width from the array shape
            height, width = results[metric].shape
            # Get the transform from the bounds and crs
            transform = from_bounds(
                west=bounds[0],
                south=bounds[1],
                east=bounds[2],
                north=bounds[3],
                width=width,
                height=height
            )
            output_file = os.path.join(outdir, f"{tile_id}_{sensor}_{veg_index.upper()}_{metric}.tif")
            # Set NaN to the nodata value
            # results[metric] = np.where(np.isnan(results[metric]), -9999, results[metric])
            with rasterio.open(output_file,
                               "w",
                               driver="GTiff",
                               height=height,
                               width=width,
                               count=1,
                               dtype=results[metric].dtype,
                               crs=crs,
                               transform=transform,
                               nodata=-9999,
                              ) as dst:
                dst.write(results[metric], 1)
                print(f"{output_file} sucessfully written.")

# ----------------------------------------------------------------------------------
# --------------------------------Plotting scripts----------------------------------
# ----------------------------------------------------------------------------------

def plot_annual_pixel_phenometrics(
        veg_index,
        sensor,
        outdir,
        tile_id,
        year):

    metrics = [
        # 'mean_vi',
        'max_vi',
        'min_vi',
        'max_doy',
        'min_doy',
        'amplitude',
        # 'greenup_doy',
        # 'dormancy_doy',
        # 'greenup_vi',
        # 'dormancy_vi',
        # 'growing_season_length',
        
        # 'greenup_threshold',
        # 'auc_full',
        # 'auc_net',
        # 'greenup_rate',
        # 'greenup_rate_doy',
        # 'senescence_rate',
        # 'senescence_rate_doy'
    ]

    n_panels = len(metrics)
    ncols    = 2
    nrows    = math.ceil(n_panels / ncols)

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(13, 4 * nrows),
                             sharex=False)
    axes = axes.flatten()

    for ax, metric in zip(axes, metrics):
        tif = os.path.join(
            outdir,
            f"{tile_id}_{sensor}_{veg_index.upper()}_{metric}_{year}.tif"
        )
        if not os.path.exists(tif):
            print(f"Missing {metric}.")
            ax.set_visible(False)
            continue
        with rasterio.open(tif) as src:
            data = src.read(1).astype(float)
            if src.nodata is not None:
                data[data == src.nodata] = np.nan
            extent = plotting_extent(src)
        im = ax.imshow(
            data,
            extent=extent,
            cmap="viridis"
        )
        ax.set_title(metric.replace("_", " ").title())
        ax.set_axis_off()
        fig.colorbar(
            im,
            ax=ax,
            fraction=0.046,
            pad=0.04
        )
    
    # hide any unused panel (if n_panels is odd)
    for ax in axes[n_panels:]:
        ax.set_visible(False)
    
    fig.suptitle(f"{tile_id} | {sensor} | {veg_index.upper()} | {year}",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    
    outfile = os.path.join(outdir, f"{tile_id}_{sensor}_{veg_index.upper()}_{year}_phenometrics.png")
    fig.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.show()
    
    print(f"Saved: {outfile}")

def build_summary_df_from_tifs(veg_index, outdir, tile_id, start_year, end_year):
    """
    Read annual phenometric GeoTIFFs and create a dataframe for plotting.

    Expected filename:
        {tile_id}_{sensor}_{veg_index}_{metric}_{year}.tif

    Each TIFF contains one year's phenometric raster.

    Returns:
        DataFrame with:
            sensor
            year
            metric columns
    """

    metrics = [
        'mean_vi',
        'max_vi',
        'min_vi',
        'max_doy',
        'min_doy',
        'amplitude',
        'greenup_doy',
        'dormancy_doy',
        'greenup_vi',
        'dormancy_vi',
        'growing_season_length',
        # 'greenup_threshold',
        # 'auc_full',
        # 'auc_net',
        # 'greenup_rate',
        # 'greenup_rate_doy',
        # 'senescence_rate',
        # 'senescence_rate_doy'
    ]

    rows = []
    
    for sensor in ["HLS", "MODIS"]:
        for year in range(start_year, end_year+1):
            row = {
                "sensor": sensor,
                "year": year
            }
    
            for metric in metrics:
                tif = os.path.join(outdir, f"{tile_id}_{sensor}_{veg_index.upper()}_{metric}_{year}.tif")

                if not os.path.exists(tif):
                    row[metric] = np.nan
                    continue
                    
                with rasterio.open(tif) as src:
                    data = src.read(1).astype(float)
                    
                    if src.nodata is not None:
                        data[data == src.nodata] = np.nan

                    # Get the annual average for each phenometric
                    row[metric] = np.nanmean(data)
                    # Get standard deviation for each phenometric
                    row[f"{metric}_std"] = np.nanstd(data)
            
            rows.append(row)
    
    summary_df = pd.DataFrame(rows)
    return summary_df

def plot_phenology_summary(summary_df, veg_index, outdir, tile_id):
    """
    Point plot: year on x-axis, one panel per metric,
    HLS = steelblue circles, MODIS = darkorange squares.
    """

    vi_upper = veg_index.upper()

    metrics = [
        # ('max_vi', f"{vi_upper} Annual Max",         "VI"),
        # ('min_vi', f"{vi_upper} Annual Min",         "VI"),
        # ('max_doy', "DOY of Annual Max",           "DMax"),
        # ('min_doy', "DOY of Annual Min",           "DMin"),
        # ('amplitude', f"{vi_upper} Range",           "VI"),
        ('greenup_doy', "Start of Season (SOS)",    "DOY"),
        ('dormancy_doy', "End of Season (EOS)",     "DOY"),
        ('greenup_vi', f"{vi_upper} at SOS",         "VI"),
        ('dormancy_vi', f"{vi_upper} at EOS",        "VI"),
        # ('growing_season_length', "Season Length", "Days"),
    ]

    n_panels = len(metrics)
    ncols    = 2
    nrows    = math.ceil(n_panels / ncols)

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(13, 4 * nrows),
                             sharex=False)
    axes = axes.flatten()

    sensor_style = {
        "HLS": dict(color="steelblue", marker="o", zorder=3),
        "MODIS": dict(color="darkorange", marker="s", zorder=3),
    }

    for ax, (col, title, ylabel) in zip(axes, metrics):
        for sensor, style in sensor_style.items():
            sub = summary_df[summary_df["sensor"] == sensor].dropna(subset=[col])
            if sub.empty:
                continue
            ax.scatter(sub["year"], sub[col],
                       label=sensor, s=60,
                       **style)
            # connect points with a thin line to help readability
            ax.plot(sub["year"], sub[col],
                    color=style["color"], linewidth=0.8,
                    alpha=0.5, zorder=2)

        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Year")
        ax.set_ylabel(ylabel)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)

    # hide any unused panel (if n_panels is odd)
    for ax in axes[n_panels:]:
        ax.set_visible(False)

    fig.suptitle(f"{tile_id}  |  {vi_upper} Annual Phenology Summary",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()

    png_path = os.path.join(outdir, f"{tile_id}_{vi_upper}_annual_phenology_summary.png")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Plot saved: {png_path}")

def raster_pearson_r(hls_tif, modis_tif, resampling=Resampling.average):
    with rasterio.open(hls_tif) as hls, rasterio.open(modis_tif) as modis:
        # Create an output array with the exact MODIS grid
        hls_resampled = np.full((modis.height, modis.width), np.nan, dtype=np.float32)

        reproject(source=rasterio.band(hls, 1), destination=hls_resampled,
                  src_transform=hls.transform, src_crs=hls.crs,
                  src_nodata=hls.nodata, dst_transform=modis.transform,
                  dst_crs=modis.crs, dst_nodata=np.nan, resampling=resampling)

        # Read MODIS
        modis_data = modis.read(1, masked=True)

        # Valid pixels in both rasters
        mask = (np.isfinite(hls_resampled) & ~modis_data.mask & np.isfinite(modis_data.data))

        # print("HLS resampled:")
        # print(np.nanmean(hls_resampled))
        # print("MODIS:")
        # print(np.nanmean(modis_data.data[~modis_data.mask]))
        # print("Valid paired pixels:", mask.sum())

        x = hls_resampled[mask]
        y = modis_data.data[mask]

        # Need at least 2 observations
        if len(x) < 2:
            return np.nan

        # Pearson r
        r = np.corrcoef(x, y)[0, 1]

        return r

def create_correlation_dfs(veg_index, tiles, start_year, end_year, metrics):
    vi_upper = veg_index.upper()
    
    correlation_dfs = []

    for tile_id in tiles:
        outdir = f"hls_modis_comparisons/{tile_id}"
        rows = []
        for year in range(start_year, end_year+1):
            row = {
                "year": year
            }
            
            for metric in metrics:
                hls_tif = os.path.join(outdir, f"{tile_id}_HLS_{vi_upper}_{metric[0]}_{year}.tif")
                modis_tif = os.path.join(outdir, f"{tile_id}_MODIS_{vi_upper}_{metric[0]}_{year}.tif")
        
                if not os.path.exists(hls_tif) or not os.path.exists(modis_tif):
                    row[metric[0]] = np.nan
                    continue
                
                row[metric[0]] = raster_pearson_r(
                    hls_tif,
                    modis_tif
                )
            
            rows.append(row)
            
        correlation_df = pd.DataFrame(rows)
        correlation_dfs.append(correlation_df)

    return correlation_dfs

def plot_line_plot(veg_index, metrics, summary_dfs, correlation_dfs, sites):
    vi_upper = veg_index.upper()
    n_panels = len(metrics)
    ncols    = len(metrics) + 1 # 2
    nrows    = len(summary_dfs) # math.ceil(n_panels / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(21, 4 * nrows), sharex=False, squeeze=False)
    axes = np.atleast_2d(axes)
    
    sensor_style = {
        "HLS": dict(color="steelblue", marker="o", zorder=3),
        "MODIS": dict(color="darkorange", marker="s", zorder=3),
    }
    
    for df_idx, summary_df in enumerate(summary_dfs):
        for metric_idx, (col, title, ylabel, std_col) in enumerate(metrics):
            ax = axes[df_idx, metric_idx]
            
            for sensor, style in sensor_style.items():
                sub = summary_df[summary_df["sensor"] == sensor].dropna(subset=[col, std_col])
                if sub.empty:
                    continue
                
                # Line plot
                ax.scatter(sub["year"], sub[col],
                           label=sensor, s=60,
                           **style)
                # connect points with a thin line to help readability
                ax.plot(sub["year"], sub[col],
                        color=style["color"], linewidth=0.8,
                        alpha=0.5, zorder=2)
                # Show std
                ax.fill_between(
                    sub["year"],
                    sub[col] - sub[std_col],
                    sub[col] + sub[std_col],
                    color=style["color"],
                    alpha=0.2,
                    linewidth=0,
                    zorder=1
                )
    
            ax.set_title(f"{title} ± 1σ", fontsize=11)
            ax.set_xlabel("Year")
            ax.set_ylabel(ylabel)
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
            ax.legend(fontsize=8, loc="best")
            ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)

        # The correlations is set to use your first 2 metrics, since it's assumed you're only
        # looking at 2 metrics
        corr_df = correlation_dfs[df_idx]
        ax_corr = axes[df_idx, 2]
        
        ax_corr.plot(corr_df["year"], corr_df[metrics[0][0]],
                     color="steelblue", marker="o",
                     linewidth=0.8, label=metrics[0][1])
        
        ax_corr.plot(corr_df["year"], corr_df[metrics[1][0]],
                     color="darkorange", marker="s",
                     linewidth=0.8, label=metrics[1][1])
        
        ax_corr.set_title("HLS–MODIS Correlation", fontsize=11)
        ax_corr.set_xlabel("Year")
        ax_corr.set_ylabel("Pearson's r")
        ax_corr.set_ylim(-1, 1)
        ax_corr.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax_corr.grid(True, linestyle="--", linewidth=0.4, alpha=0.6) 
        ax_corr.legend(fontsize=8, loc="best")
    
    # Label rows
    for row_idx, site in enumerate(sites):
        axes[row_idx, 0].annotate(site, xy=(-0.25, 0.5), xycoords="axes fraction",
                                  rotation=90, va="center", ha="center",
                                  fontsize=11, fontweight="bold")
    
    fig.suptitle(f"{vi_upper} Annual Phenology Summary",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    
    png_path = os.path.join("hls_modis_comparisons/", f"{vi_upper}_annual_phenology_summary.png")
    fig.savefig(png_path, dpi=500, bbox_inches="tight")
    plt.show()
    print(f"Plot saved: {png_path}")

def plot_bar_plot(veg_index, metrics, summary_dfs, correlation_dfs, sites):
    vi_upper = veg_index.upper()
    n_panels = len(metrics)
    ncols    = len(metrics) + 1 # 2
    nrows    = len(summary_dfs) # math.ceil(n_panels / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(21, 4 * nrows), sharex=False, squeeze=False)
    axes = np.atleast_2d(axes)
    width = 0.35
    
    sensor_style = {
        "HLS": dict(color="steelblue", marker="o", zorder=3),
        "MODIS": dict(color="darkorange", marker="s", zorder=3),
    }
    
    for df_idx, summary_df in enumerate(summary_dfs):
        for metric_idx, (col, title, ylabel, std_col) in enumerate(metrics):
            ax = axes[df_idx, metric_idx]
            
            for sensor, style in sensor_style.items():
                sub = summary_df[summary_df["sensor"] == sensor].dropna(subset=[col, std_col])
                if sub.empty:
                    continue
                
                # Bar plot
                x = np.arange(len(sub))
                offset = -width / 2 if sensor == "HLS" else width / 2
                ax.bar(x + offset, sub[col], width=width,
                       color=style["color"], yerr=sub[std_col],
                       capsize=3, label=sensor)
    
            ax.set_title(f"{title} ± 1σ", fontsize=11)
            ax.set_xlabel("Year")
            ax.set_ylabel(ylabel)
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
            ax.legend(fontsize=8, loc="best")
            ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)

        # The correlations is set to use your first 2 metrics, since it's assumed you're only
        # looking at 2 metrics
        corr_df = correlation_dfs[df_idx]
        ax_corr = axes[df_idx, 2]
        
        ax_corr.plot(corr_df["year"], corr_df[metrics[0][0]],
                     color="steelblue", marker="o",
                     linewidth=0.8, label=metrics[0][1])
        
        ax_corr.plot(corr_df["year"], corr_df[metrics[1][0]],
                     color="darkorange", marker="s",
                     linewidth=0.8, label=metrics[1][1])
        
        ax_corr.set_title("HLS–MODIS Correlation", fontsize=11)
        ax_corr.set_xlabel("Year")
        ax_corr.set_ylabel("Pearson's r")
        ax_corr.set_ylim(-1, 1)
        ax_corr.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax_corr.grid(True, linestyle="--", linewidth=0.4, alpha=0.6) 
        ax_corr.legend(fontsize=8, loc="best")
    
    # Label rows
    for row_idx, site in enumerate(sites):
        axes[row_idx, 0].annotate(site, xy=(-0.25, 0.5), xycoords="axes fraction",
                                  rotation=90, va="center", ha="center",
                                  fontsize=11, fontweight="bold")
    
    fig.suptitle(f"{vi_upper} Annual Phenology Summary",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    
    png_path = os.path.join("hls_modis_comparisons/", f"{vi_upper}_annual_phenology_summary.png")
    fig.savefig(png_path, dpi=500, bbox_inches="tight")
    plt.show()
    print(f"Plot saved: {png_path}")

def plot_box_and_whiskers_plot(veg_index, metrics, tiles, start_year, end_year, correlation_dfs, sites):
    # Box and whiskers plot version
    vi_upper = veg_index.upper()
    n_panels = len(metrics)
    ncols    = len(metrics) + 1 # 2
    nrows    = len(tiles) # math.ceil(n_panels / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(21, 4 * nrows), sharex=False, squeeze=False)
    axes = np.atleast_2d(axes)
    
    for tile_idx, (tile_id) in enumerate(tiles):
        outdir = f"hls_modis_comparisons/{tile_id}"
        for metric_idx, (metric, title, ylabel, metric_std) in enumerate(metrics):
            ax = axes[tile_idx, metric_idx]
            sensor_values = {
                "HLS": {},
                "MODIS": {}
            }
            for sensor in ["HLS", "MODIS"]:
                for year in range(start_year, end_year+1):
                    tif = os.path.join(outdir, f"{tile_id}_{sensor}_{vi_upper}_{metric}_{year}.tif")
                    if not os.path.exists(tif):
                        continue
                    
                    with rasterio.open(tif) as src:
                        data = src.read(1)
                    
                    # Keep only finite values
                    values = data[np.isfinite(data)]
                    
                    if len(values) == 0:
                        continue
    
                    sensor_values[sensor][year] = values
    
            year_list = sorted(set(sensor_values["HLS"]) | set(sensor_values["MODIS"]))
    
            positions_hls = []
            positions_modis = []
            hls_data = []
            modis_data = []
    
            for i, year in enumerate(year_list):
                center = i + 1
    
                if year in sensor_values["HLS"]:
                    positions_hls.append(center - 0.18)
                    hls_data.append(sensor_values["HLS"][year])
    
                if year in sensor_values["MODIS"]:
                    positions_modis.append(center + 0.18)
                    modis_data.append(sensor_values["MODIS"][year])
    
            if hls_data:
                bp_hls = ax.boxplot(hls_data, positions=positions_hls, widths=0.30,
                                    patch_artist=True, showfliers=False,
                                    medianprops=dict(color="black", linewidth=1),
                                    whiskerprops=dict(color="steelblue"),
                                    capprops=dict(color="steelblue"), label="HLS")
    
                for box in bp_hls["boxes"]:
                    box.set_facecolor("steelblue")
                    box.set_alpha(0.65)
    
            if modis_data:
                bp_modis = ax.boxplot(modis_data, positions=positions_modis, widths=0.30,
                                      patch_artist=True, showfliers=False,
                                      medianprops=dict(color="black", linewidth=1),
                                      whiskerprops=dict(color="darkorange"),
                                      capprops=dict(color="darkorange"), label="MODIS")
    
                for box in bp_modis["boxes"]:
                    box.set_facecolor("darkorange")
                    box.set_alpha(0.65)
                    
            ax.set_title(title, fontsize=11)
            ax.set_xlabel("Year")
            ax.set_ylabel(ylabel)
            ax.set_xticks(range(1, len(year_list) + 1))
            ax.set_xticklabels(year_list)
            ax.legend(fontsize=8, loc="best")
            ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)
    
        # The correlations is set to use your first 2 metrics, since it's assumed you're only
        # looking at 2 metrics
        corr_df = correlation_dfs[tile_idx]
        ax_corr = axes[tile_idx, 2]
        
        ax_corr.plot(corr_df["year"], corr_df[metrics[0][0]],
                     color="steelblue", marker="o",
                     linewidth=0.8, label=metrics[0][1])
        
        ax_corr.plot(corr_df["year"], corr_df[metrics[1][0]],
                     color="darkorange", marker="s",
                     linewidth=0.8, label=metrics[1][1])
        
        ax_corr.set_title("HLS–MODIS Correlation", fontsize=11)
        ax_corr.set_xlabel("Year")
        ax_corr.set_ylabel("Pearson's r")
        ax_corr.set_ylim(-1, 1)
        ax_corr.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax_corr.grid(True, linestyle="--", linewidth=0.4, alpha=0.6) 
        ax_corr.legend(fontsize=8, loc="best")
    
    
    # Label rows
    for row_idx, site in enumerate(sites):
        axes[row_idx, 0].annotate(site, xy=(-0.25, 0.5), xycoords="axes fraction",
                                  rotation=90, va="center", ha="center",
                                  fontsize=11, fontweight="bold")
    
    
    fig.suptitle(f"{vi_upper} Annual Phenology Summary",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    
    png_path = os.path.join("hls_modis_comparisons/", f"{vi_upper}_annual_phenology_summary.png")
    fig.savefig(png_path, dpi=500, bbox_inches="tight")
    plt.show()
    print(f"Plot saved: {png_path}")