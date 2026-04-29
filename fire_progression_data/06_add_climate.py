"""
06_add_climate.py
=================
Adds ERA5 weather variables to fire grid shapefiles.

Variables extracted per grid cell per acquisition hour:
    - 2m temperature (°C), dewpoint temperature (°C)
    - Relative humidity (%), vapour pressure deficit (kPa)
    - Sea surface temperature (°C)
    - Wind speed (m/s) and direction (degrees from North)
    - Wind gust (m/s)
    - Total precipitation (m, hourly increment)
    - Evaporation and potential evaporation (m, hourly increment)
    - Convective precipitation (m, hourly increment)
    - Convective rain rate (mm/hr)
    - Surface pressure (Pa)
    - Mean sea level pressure (Pa)

Pipeline:
    1. Load fire grid shapefiles
    2. Open ERA5 NetCDF files per variable per year-month
    3. Extract values at each grid centroid (lon/lat) and nearest hour
    4. Save enriched grid shapefile
"""

import os
import gc
import glob

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr

from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths 
# ---------------------------------------------------------------------------

ERA5_FOLDERS = {
    "sumatra":    "ERA5_Climate/Sumatra",
    "kalimantan": "ERA5_Climate/Kalimantan",
}

INPUT_PATHS = {
    "sumatra":    "data_preprocess/fire_event_shp_modis_af/attributes/mapbiomas_c4_ver/v4_500m_60cl/4_grids_ndvi_dem/sumatra",
    "kalimantan": "data_preprocess/fire_event_shp_modis_af/attributes/mapbiomas_c4_ver/v4_500m_60cl/4_grids_ndvi_dem/kalimantan",
}

OUTPUT_PATHS = {
    "sumatra":    "data_preprocess/fire_event_shp_modis_af/attributes/mapbiomas_c4_ver/v4_500m_60cl/5_grids_climate/sumatra",
    "kalimantan": "data_preprocess/fire_event_shp_modis_af/attributes/mapbiomas_c4_ver/v4_500m_60cl/5_grids_climate/kalimantan",
}

YEAR_RANGE = range(2001, 2025)

ERA5_VARIABLES = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_dewpoint_temperature",
    "2m_temperature",
    "mean_sea_level_pressure",
    "sea_surface_temperature",
    "surface_pressure",
    "total_precipitation",
    "instantaneous_10m_wind_gust",
    "evaporation",
    "potential_evaporation",
    "convective_precipitation",
    "convective_rain_rate",
]

# ---------------------------------------------------------------------------
# ERA5
# ---------------------------------------------------------------------------

def _open_era5(era_folder, variable, year, month):
    """Open a single ERA5 monthly NetCDF file."""
    month_str = f"{month:02d}"
    fp = os.path.join(era_folder, variable, f"era5_{variable}_{year}_{month_str}.nc")
    return xr.open_dataset(fp).rename({"valid_time": "time"})


def _sel_nearest(ds, var, lon, lat, time):
    """Extract scalar value at nearest grid point and nearest hour."""
    return (
        ds[var]
        .sel(longitude=lon, latitude=lat, time=time.round("1h"), method="nearest")
        .values.item()
    )


def _hourly_diff(ds, var, lon, lat, time):
    """Convert accumulated variable to hourly increment, then extract value."""
    hourly = ds[var].diff("time", label="upper").fillna(0)
    return hourly.sel(
        longitude=lon, latitude=lat, time=time.round("1h"), method="nearest"
    ).values.item()


def add_ym_to_gdf(gdf):
    """Ensure year and month columns exist using n_ACQ_DATE."""
    if "year" in gdf.columns and "month" in gdf.columns:
        return gdf
    gdf = gdf.copy()
    gdf["n_ACQ_DATE"] = pd.to_datetime(gdf["n_ACQ_DATE"], errors="coerce")
    gdf["year"]  = gdf["n_ACQ_DATE"].dt.year
    gdf["month"] = gdf["n_ACQ_DATE"].dt.month
    return gdf


def add_weather_variables(gdf, era_folder):
    """
    Extract all ERA5 weather variables for each grid row.
    Opens ERA5 files once per unique year-month combination.
    """
    for year, month in gdf[["year", "month"]].drop_duplicates().itertuples(index=False, name=None):
        idx   = (gdf["year"] == year) & (gdf["month"] == month)
        gdf_m = gdf[idx]

        # Open all ERA5 datasets for this month
        ds_tp   = _open_era5(era_folder, "total_precipitation",         year, month)
        ds_temp = _open_era5(era_folder, "2m_temperature",              year, month)
        ds_dew  = _open_era5(era_folder, "2m_dewpoint_temperature",     year, month)
        ds_u10  = _open_era5(era_folder, "10m_u_component_of_wind",     year, month)
        ds_v10  = _open_era5(era_folder, "10m_v_component_of_wind",     year, month)
        ds_gust = _open_era5(era_folder, "instantaneous_10m_wind_gust", year, month)
        ds_evap = _open_era5(era_folder, "evaporation",                 year, month)
        ds_pev  = _open_era5(era_folder, "potential_evaporation",       year, month)
        ds_cp   = _open_era5(era_folder, "convective_precipitation",    year, month)
        ds_crr  = _open_era5(era_folder, "convective_rain_rate",        year, month)
        ds_sp   = _open_era5(era_folder, "surface_pressure",            year, month)
        ds_msl  = _open_era5(era_folder, "mean_sea_level_pressure",     year, month)
        ds_sst  = _open_era5(era_folder, "sea_surface_temperature",     year, month)
        ds_wind = xr.merge([ds_u10, ds_v10])

        for i, row in gdf_m.iterrows():
            lon, lat, t = row["lon"], row["lat"], row["n_ACQ_DATE"]

            # Temperature, dewpoint, relative humidity, vapour pressure deficit
            t2m = _sel_nearest(ds_temp, "t2m", lon, lat, t) - 273.15 # To Celcius
            d2m = _sel_nearest(ds_dew,  "d2m", lon, lat, t) - 273.15 # To Celcius
            es  = 0.6108 * np.exp((17.27 * t2m) / (t2m + 237.3))
            ea  = 0.6108 * np.exp((17.27 * d2m) / (d2m + 237.3))
            gdf.at[i, "t2m"] = t2m
            gdf.at[i, "d2m"] = d2m
            gdf.at[i, "rh"]  = (ea / es) * 100.0
            gdf.at[i, "vpd"] = es - ea

            # Sea surface temperature
            gdf.at[i, "sst"] = _sel_nearest(ds_sst, "sst", lon, lat, t) - 273.15 # To Celcius

            # Wind speed and direction
            u = _sel_nearest(ds_wind, "u10", lon, lat, t)
            v = _sel_nearest(ds_wind, "v10", lon, lat, t)
            gdf.at[i, "wind_speed"] = np.sqrt(u**2 + v**2)
            gdf.at[i, "dir_deg"]    = (np.arctan2(u, v) * 180 / np.pi + 180) % 360
            gdf.at[i, "gust"]       = _sel_nearest(ds_gust, "i10fg", lon, lat, t)

            # Precipitation and evaporation (hourly increments)
            gdf.at[i, "tp_hourly"]          = _hourly_diff(ds_tp,   "tp",  lon, lat, t)
            gdf.at[i, "ev_h"]               = _hourly_diff(ds_evap,  "e",   lon, lat, t)
            gdf.at[i, "pev_hourly"]         = _hourly_diff(ds_pev,   "pev", lon, lat, t)
            gdf.at[i, "conv_precip_hourly"] = _hourly_diff(ds_cp,    "cp",  lon, lat, t)
            gdf.at[i, "conv_rain_rate"]     = _sel_nearest(ds_crr,   "crr", lon, lat, t) * 3600.0 # To mm/hr

            # Pressure
            gdf.at[i, "sp_pa"]   = _sel_nearest(ds_sp,  "sp",  lon, lat, t)
            gdf.at[i, "mslp_pa"] = _sel_nearest(ds_msl, "msl", lon, lat, t)

    return gdf


# ---------------------------------------------------------------------------
# Per-fire processor
# ---------------------------------------------------------------------------

def process_fire(shp, era_folder, year):
    """Add all ERA5 weather variables to one fire grid shapefile."""
    gdf = gpd.read_file(shp)
    if gdf.empty:
        return None

    gdf["n_ACQ_DATE"] = pd.to_datetime(gdf["n_ACQ_DATE"], errors="coerce")
    gdf = add_ym_to_gdf(gdf)
    gdf = add_weather_variables(gdf, era_folder)

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

        for year in YEAR_RANGE:
            shp_list = glob.glob(os.path.join(INPUT_PATHS[province], f"*fire_{year}_*.shp"))
            if not shp_list:
                continue
            print(f"\n  Year {year}: {len(shp_list)} fires")

            for shp in tqdm(shp_list, desc=f"Year {year}", unit="fire", leave=False):
                try:
                    result = process_fire(shp, ERA5_FOLDERS[province], year)
                    if result is not None:
                        gdf, fire_id, year_name = result
                        out_path = os.path.join(out_folder, f"fire_{year_name}_id_{fire_id}_grid.shp")
                        gdf.to_file(out_path)
                except Exception as e:
                    print(f"  Failed {os.path.basename(shp)}: {e}")
                finally:
                    gc.collect()

    print("\n Add weather complete.")
