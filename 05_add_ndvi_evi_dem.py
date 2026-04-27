"""
04_add_ndvi_evi_dem.py
======================
Adds MODIS NDVI/EVI and NASADEM elevation to fire grid shapefiles.

Pipeline:
    1. Load fire grid shapefiles (output of 03_create_grids.py)
    2. Extract MODIS NDVI and EVI from nearest 16-day composite (NetCDF)
    3. Apply MODIS scale factor (÷ 0.0001)
    4. Merge NASADEM elevation by grid_id
    5. Save enriched grid shapefile
"""

import os
import gc
import glob

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import cftime

from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

NDVI_FOLDERS = {
    "sumatra":    "MODIS_NDVI/Sumatra",
    "kalimantan": "MODIS_NDVI/Kalimantan",
}

DEM_PATHS = {
    "sumatra":    "NASADEM/sumatra_dem.shp",
    "kalimantan": "NASADEM/kalimantan_dem.shp",
}

INPUT_PATHS = {
    "sumatra":    "data_preprocess/fire_event_shp_modis_af/attributes/mapbiomas_c4_ver/v4_500m_60cl/3_create_grids/sumatra",
    "kalimantan": "data_preprocess/fire_event_shp_modis_af/attributes/mapbiomas_c4_ver/v4_500m_60cl/3_create_grids/kalimantan",
}

OUTPUT_PATHS = {
    "sumatra":    "data_preprocess/fire_event_shp_modis_af/attributes/mapbiomas_c4_ver/v4_500m_60cl/4_grids_ndvi_dem/sumatra",
    "kalimantan": "data_preprocess/fire_event_shp_modis_af/attributes/mapbiomas_c4_ver/v4_500m_60cl/4_grids_ndvi_dem/kalimantan",
}

YEAR_RANGE = range(2001, 2025)

# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

def add_ndvi_evi(gdf, ds_path):
    """
    Extract MODIS NDVI and EVI for each grid row from a NetCDF file.
    Selects the nearest 16-day composite to n_ACQ_DATE.
    """
    gdf = gdf.copy()
    gdf["ACQ_DATE_CF"] = gdf["n_ACQ_DATE"].apply(
        lambda x: cftime.DatetimeJulian(x.year, x.month, x.day)
    )
    ds   = xr.open_dataset(ds_path)
    ndvi = ds["_500m_16_days_NDVI"].where(ds["_500m_16_days_NDVI"] != -3000)
    evi  = ds["_500m_16_days_EVI"].where(ds["_500m_16_days_EVI"]  != -3000)

    lats     = xr.DataArray(gdf["lat"].values,         dims="points")
    lons     = xr.DataArray(gdf["lon"].values,         dims="points")
    dates_cf = xr.DataArray(gdf["ACQ_DATE_CF"].values, dims="points")
    t_sel    = ndvi.time.sel(time=dates_cf, method="nearest")

    gdf["ndvi"]      = ndvi.sel(time=t_sel, lat=lats, lon=lons, method="nearest").values
    gdf["evi"]       = evi.sel( time=t_sel, lat=lats, lon=lons, method="nearest").values
    gdf["ndvi_date"] = pd.to_datetime(t_sel.values.astype(str))
    return gdf


def add_dem(gdf, dem_gdf):
    """Merge elevation from DEM shapefile by grid_id."""
    return gdf.merge(dem_gdf[["grid_id", "elevation"]], on="grid_id", how="left")


def process_fire(shp, dem_gdf, ndvi_path, year):
    """Add NDVI, EVI (scaled), and elevation to one fire grid shapefile."""
    gdf = gpd.read_file(shp)
    if gdf.empty:
        return None

    gdf["n_ACQ_DATE"] = pd.to_datetime(gdf["n_ACQ_DATE"], errors="coerce")

    gdf = add_ndvi_evi(gdf, ndvi_path)
    gdf = add_dem(gdf, dem_gdf)

    fire_id   = gdf["fire_id"].iloc[0]
    year_name = gdf["n_ACQ_DATE"].dt.year.iloc[0]
    return gdf, fire_id, year_name


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    for province in ["sumatra", "kalimantan"]:
        out_folder = OUTPUT_PATHS[province]
        os.makedirs(out_folder, exist_ok=True)

        print(f"\n{'='*50}")
        print(f"Province: {province.upper()}")
        print(f"{'='*50}")

        print("  Loading DEM ...")
        dem_gdf = gpd.read_file(DEM_PATHS[province])
        if "grid_id" not in dem_gdf.columns:
            dem_gdf = dem_gdf.reset_index(names="grid_id")

        for year in YEAR_RANGE:
            shp_list = glob.glob(os.path.join(INPUT_PATHS[province], f"*fire_{year}_*.shp"))
            if not shp_list:
                continue

            ndvi_path = os.path.join(NDVI_FOLDERS[province], f"MODIS_NDVI_{year}.nc")
            if not os.path.exists(ndvi_path):
                print(f"  NDVI missing for {year}, skipping.")
                continue

            print(f"\n  Year {year}: {len(shp_list)} fires")

            for shp in tqdm(shp_list, desc=f"Year {year}", unit="fire", leave=False):
                try:
                    result = process_fire(shp, dem_gdf, ndvi_path, year)
                    if result is not None:
                        gdf, fire_id, year_name = result
                        out_path = os.path.join(out_folder, f"fire_{year_name}_id_{fire_id}_grid.shp")
                        gdf.to_file(out_path)
                except Exception as e:
                    print(f"  Failed {os.path.basename(shp)}: {e}")
                finally:
                    gc.collect()

    print("\nAdd NDVI, EVI, and DEM complete.")
