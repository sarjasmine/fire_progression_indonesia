"""
0_preprocess_lulc.py
======================
Preprocesses MapBiomas LULC data for Sumatra and Kalimantan.
Based on:
    0__Edit_LULC_bounds.ipynb     — clip Indonesia-wide TIFFs to province bounds
    0__resampling_lulc.ipynb      — resample 30m LULC to 500m (MODIS-matched)
    1__Export_LULC-v2.ipynb       — extract majority LULC per grid cell and fill NoData


Pipeline:
    Step A — Clip Indonesia-wide MapBiomas TIFFs to province boundaries
    Step B — Resample clipped 30m TIFFs to 500m using mode resampling
             + fill NoData pixels using nearest-neighbour
    Step C — Extract majority LULC per province grid cell using zonal stats
             + fill remaining NaN cells via KDTree nearest neighbour
             + save one shapefile per province per year

Inputs:
    - MapBiomas Indonesia TIFFs : MapBiomas_LULC/indonesia_coverage_{year}.tif
    - Province boundaries       : INA_GEOPORTAL/ADMIN_edited/{Sumatra,Kalimantan}_island_dissolved.shp
    - Province grids            : data_preprocess/{sumatra,kalimantan}_grids_ba.shp

Outputs:
    - Clipped TIFFs             : MapBiomas_LULC/{sumatra,kalimantan}_coverage_{year}.tif
    - Resampled TIFFs           : MapBiomas_LULC/resampled_500m/{sumatra,kalimantan}/{Province}_coverage_{year}.tif
    - LULC grid shapefiles      : MapBiomas_LULC/grid_lulc_removed_nan_lulc/mapbiomas_{Province}_{year}.shp
"""

import os
import gc
import glob

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.enums import Resampling
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
from rasterstats import zonal_stats
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths 
# ---------------------------------------------------------------------------

MAPBIOMAS_FOLDER  = "MapBiomas_LULC"
SHP_FOLDER        = "INA_GEOPORTAL/ADMIN_edited"

PROVINCE_SHP_PATHS = {
    "Sumatra":    os.path.join(SHP_FOLDER, "Sumatra_island_dissolved.shp"),
    "Kalimantan": os.path.join(SHP_FOLDER, "Kalimantan_island_dissolved.shp"),
}

GRID_PATHS = {
    "Sumatra":    "data_preprocess/sumatra_grids_ba.shp",
    "Kalimantan": "data_preprocess/kalimantan_grids_ba.shp",
}

MAPBIOMAS_RESAMPLED_FOLDERS = {
    "Sumatra":    os.path.join(MAPBIOMAS_FOLDER, "resampled_500m", "sumatra"),
    "Kalimantan": os.path.join(MAPBIOMAS_FOLDER, "resampled_500m", "kalimantan"),
}

OUTPUT_LULC_FOLDER = "MapBiomas_LULC/grid_lulc_removed_nan_lulc"

YEAR_RANGE = range(2001, 2023)

# ---------------------------------------------------------------------------
# LULC reference (MapBiomas Indonesia Collection 4)
# ---------------------------------------------------------------------------

LULC_INDEX = {
    0:  "No Data",
    1:  "Forest",
    3:  "Forest Formation",
    5:  "Mangrove",
    9:  "Pulpwood Plantation",
    10: "Non Forest Natural Formation",
    13: "Non Forest Natural Vegetation",
    18: "Agriculture",
    21: "Other Agriculture",
    22: "Non Vegetated",
    24: "Urban Area",
    25: "Other Non Vegetation",
    26: "Water",
    27: "Non Observed",
    30: "Mining Pit",
    31: "Aquaculture",
    33: "River, Lake, Ocean",
    35: "Oil Palm",
    40: "Rice Paddy",
    76: "Peat Swamp Forest",
}

# ---------------------------------------------------------------------------
# Step A: Clip Indonesia-wide TIFFs to province boundaries
# ---------------------------------------------------------------------------

def clip_raster_to_province(province, year):
    """
    Clip Indonesia-wide MapBiomas TIFF to a province boundary and save.

    Parameters
    ----------
    province : str
        'Sumatra' or 'Kalimantan'
    year : int

    Output
    ------
    MapBiomas_LULC/{province_lower}_coverage_{year}.tif
    """
    raster_path = os.path.join(MAPBIOMAS_FOLDER, f"indonesia_coverage_{year}.tif")
    output_path = os.path.join(MAPBIOMAS_FOLDER, f"{province.lower()}_coverage_{year}.tif")

    if not os.path.exists(raster_path):
        print(f"  Raster not found for {year}: {raster_path}")
        return

    shp = gpd.read_file(PROVINCE_SHP_PATHS[province])

    with rasterio.open(raster_path) as src:
        if shp.crs != src.crs:
            shp = shp.to_crs(src.crs)
        out_image, out_transform = mask(src, shp.geometry, crop=True)
        out_meta = src.meta.copy()
        out_meta.update({
            "driver":    "GTiff",
            "height":    out_image.shape[1],
            "width":     out_image.shape[2],
            "transform": out_transform,
            "compress":  "lzw",
            "tiled":     True,
            "dtype":     src.dtypes[0],
        })

    with rasterio.open(output_path, "w", **out_meta) as dest:
        dest.write(out_image)

    print(f"  Clipped: {output_path}")


def run_step_a():
    """Clip all years for both provinces."""
    for province in ["Sumatra", "Kalimantan"]:
        print(f"\n{'='*50}")
        print(f"Step A — Clipping: {province.upper()}")
        print(f"{'='*50}")
        for year in YEAR_RANGE:
            clip_raster_to_province(province, year)


# ---------------------------------------------------------------------------
# Step B: Resample clipped 30m TIFFs to 500m
# ---------------------------------------------------------------------------

def fill_nodata_nearest(data, nodata):
    """Fill NoData pixels using nearest-neighbour from scipy distance transform."""
    fill_mask = data == nodata
    if not fill_mask.any():
        return data
    _, indices = distance_transform_edt(fill_mask, return_indices=True)
    return data[tuple(indices)]


def resample_then_fill(input_fp, output_fp, scale_factor=500 / 30, nodata=255):
    """
    Resample a 30m LULC raster to 500m using mode resampling,
    then fill any NoData pixels using nearest-neighbour.

    Parameters
    ----------
    input_fp : str
        Path to clipped 30m TIFF.
    output_fp : str
        Path to write resampled 500m TIFF.
    scale_factor : float
        Ratio of output to input resolution (default 500/30).
    nodata : int
        NoData value to fill (default 255 for MapBiomas uint8).
    """
    with rasterio.open(input_fp) as src:
        new_height = int(src.height / scale_factor)
        new_width  = int(src.width  / scale_factor)

        data_resampled = src.read(
            1,
            out_shape=(new_height, new_width),
            resampling=Resampling.mode,
        )
        data_filled = fill_nodata_nearest(data_resampled, nodata)

        new_transform = src.transform * src.transform.scale(
            src.width  / data_filled.shape[1],
            src.height / data_filled.shape[0],
        )
        profile = src.profile.copy()
        profile.update({
            "transform": new_transform,
            "height":    new_height,
            "width":     new_width,
            "nodata":    nodata,
            "dtype":     data_filled.dtype,
        })

    os.makedirs(os.path.dirname(output_fp), exist_ok=True)
    with rasterio.open(output_fp, "w", **profile) as dst:
        dst.write(data_filled, 1)


def run_step_b():
    """Resample all clipped TIFFs from 30m to 500m for both provinces."""
    for province in ["Sumatra", "Kalimantan"]:
        out_folder = MAPBIOMAS_RESAMPLED_FOLDERS[province]
        os.makedirs(out_folder, exist_ok=True)
        print(f"\n{'='*50}")
        print(f"Step B — Resampling: {province.upper()}")
        print(f"{'='*50}")
        for year in YEAR_RANGE:
            input_fp  = os.path.join(MAPBIOMAS_FOLDER, f"{province}_coverage_{year}.tif")
            output_fp = os.path.join(out_folder, f"{province}_coverage_{year}.tif")
            if not os.path.exists(input_fp):
                print(f"  Missing: {input_fp}")
                continue
            resample_then_fill(input_fp, output_fp)
            print(f"  Resampled: {output_fp}")


# ---------------------------------------------------------------------------
# Step C: Extract majority LULC per grid cell + fill NaN + save shapefile
# ---------------------------------------------------------------------------

def add_grid_id(gdf):
    """Add grid_id column from index if not already present."""
    if "grid_id" not in gdf.columns:
        gdf = gdf.copy()
        gdf["grid_id"] = gdf.index.astype(int)
    return gdf


def add_lulc_mapbiomas(gdf_grid, mapbiomas_folder, province, year):
    """
    Compute majority MapBiomas LULC code per grid cell using zonal statistics.

    Parameters
    ----------
    gdf_grid : GeoDataFrame
        Province grid
    mapbiomas_folder : str
        Folder containing resampled 500m TIFFs for the province.
    province : str
        'Sumatra' or 'Kalimantan'
    year : int

    Returns
    -------
    GeoDataFrame with added 'mapbiomas' column (majority LULC code per cell).
    """
    search_pattern = os.path.join(mapbiomas_folder, f"{province}_coverage_{year}.tif")
    matching_files = glob.glob(search_pattern)

    if not matching_files:
        raise FileNotFoundError(
            f"No MapBiomas file found for {province}, year {year} in {mapbiomas_folder}"
        )

    mapbiomas_fp = matching_files[0]

    with rasterio.Env(GDAL_CACHEMAX=128):
        stats = zonal_stats(
            gdf_grid,
            mapbiomas_fp,
            stats="majority",
            geojson_out=False,
            nodata=255,
            raster_out=False,
        )

    gdf_grid = gdf_grid.copy()
    gdf_grid["mapbiomas"] = [s.get("majority") for s in stats]
    return gdf_grid


def fill_lulc_mapbiomas(gdf):
    """
    Fill NaN mapbiomas values using KDTree nearest-neighbour from valid cells.

    Parameters
    ----------
    gdf : GeoDataFrame
        Grid with 'mapbiomas' column potentially containing NaN values.

    Returns
    -------
    GeoDataFrame with NaN mapbiomas values filled.
    """
    valid   = gdf[~gdf["mapbiomas"].isna()].copy()
    missing = gdf[gdf["mapbiomas"].isna()].copy()

    if missing.empty:
        return gdf
    if valid.empty:
        print("  No valid mapbiomas values to fill from.")
        return gdf

    valid_coords   = np.array(list(zip(valid.geometry.centroid.x,   valid.geometry.centroid.y)))
    missing_coords = np.array(list(zip(missing.geometry.centroid.x, missing.geometry.centroid.y)))

    tree = cKDTree(valid_coords)
    _, indices = tree.query(missing_coords, k=1)

    gdf.loc[missing.index, "mapbiomas"] = valid.iloc[indices]["mapbiomas"].astype(int).values
    return gdf


def run_step_c():
    """Extract, fill, and save LULC grid shapefiles for all years and provinces."""
    os.makedirs(OUTPUT_LULC_FOLDER, exist_ok=True)

    for province in ["Sumatra", "Kalimantan"]:
        print(f"\n{'='*50}")
        print(f"Step C — Exporting LULC grids: {province.upper()}")
        print(f"{'='*50}")

        grid = add_grid_id(gpd.read_file(GRID_PATHS[province]))
        biomas_folder = MAPBIOMAS_RESAMPLED_FOLDERS[province]

        for year in YEAR_RANGE:
            print(f"  Processing {year} ...")
            try:
                gdf_year = add_lulc_mapbiomas(grid, biomas_folder, province, year)
                gdf_year = fill_lulc_mapbiomas(gdf_year)
                gdf_year["year"] = year

                out_fp = os.path.join(OUTPUT_LULC_FOLDER, f"mapbiomas_{province}_{year}.shp")
                gdf_year.to_file(out_fp, driver="ESRI Shapefile")
                print(f"  Saved: {out_fp}")

            except Exception as e:
                print(f"  Failed for {year}: {e}")
            finally:
                del gdf_year
                gc.collect()

        del grid
        gc.collect()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Step A: Clipping Indonesia TIFFs to province bounds ...")
    run_step_a()

    print("\nStep B: Resampling 30m → 500m ...")
    run_step_b()

    print("\nStep C: Extracting LULC per grid cell ...")
    run_step_c()

    print("\nLULC preprocessing complete.")
