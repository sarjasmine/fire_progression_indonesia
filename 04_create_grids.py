"""
04_create_grids.py
==================
Creates fire event grids by intersecting hull polygons with the province grid.

Pipeline:
    1. Load convex hull polygon shapefiles (output of 03_create_fire_hull.py)
    2. Load AF point shapefiles (output of 01_create_fire_clusters.py)
    3. Intersect hull with province grid → fire grid cells
    4. Aggregate AF point attributes onto matched grid cells
    5. Assign nearest AF point date/time to each grid cell
    6. Save one grid shapefile per fire event per year
"""

import os


import geopandas as gpd
import pandas as pd
import numpy as np
import glob

from shapely.geometry import Point, box

import cftime
from collections import Counter

import gc
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths — edit these to match your local data layout
# ---------------------------------------------------------------------------

GRID_PATHS = {
    "sumatra":    "data_preprocess/grids/sumatra_grids_ba.shp",
    "kalimantan": "data_preprocess/grids/kalimantan_grids_ba.shp",
}

HULL_INPUT_PATHS = {
    "sumatra":    "data_preprocess/fire_event_shp_modis_af/attributes/mapbiomas_c4_ver/v4_500m_60cl/2_create_alpha_hull/sumatra",
    "kalimantan": "data_preprocess/fire_event_shp_modis_af/attributes/mapbiomas_c4_ver/v4_500m_60cl/2_create_alpha_hull/kalimantan",
}

POINT_INPUT_PATHS = {
    "sumatra":    "data_preprocess/fire_event_shp_modis_af/attributes/mapbiomas_c4_ver/v4_500m_60cl/1_fire_character/sumatra",
    "kalimantan": "data_preprocess/fire_event_shp_modis_af/attributes/mapbiomas_c4_ver/v4_500m_60cl/1_fire_character/kalimantan",
}

OUTPUT_PATHS = {
    "sumatra":    "data_preprocess/fire_event_shp_modis_af/attributes/mapbiomas_c4_ver/v4_500m_60cl/3_create_grids/sumatra",
    "kalimantan": "data_preprocess/fire_event_shp_modis_af/attributes/mapbiomas_c4_ver/v4_500m_60cl/3_create_grids/kalimantan",
}

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

ALBERS_INDO_CRS = (
    "+proj=aea +lon_0=107.017 +lat_0=-0.13815 "
    "+lat_1=4.0417250000000005 +lat_2=-4.3180250000000004 "
    "+datum=WGS84 +units=m +no_defs"
)

YEAR_RANGE = range(2001, 2025)

# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

def edit_grids(gdf_grid, crs=ALBERS_INDO_CRS):
    """Reproject province grid and add grid_id index column."""
    return gdf_grid.to_crs(crs).reset_index().rename(columns={"index": "grid_id"})


def create_grid_from_hull(gdf_hull, gdf_grid_province):
    """Return grid cells that fall inside the fire hull polygon."""
    joined = gpd.sjoin(gdf_grid_province, gdf_hull, how="inner")
    return joined.drop("index_right", axis=1)


def prep_point(point_gdf, grid_gdf):
    """Aggregate AF points per grid_id and merge onto the grid."""
    af_agg = (
        point_gdf.groupby("grid_id")
        .agg(af_type=("af_type", "first"), af_ig_=("index_af", "count"))
        .reset_index()
    )
    grids = grid_gdf.merge(af_agg, on="grid_id", how="left")
    grids["af_type"] = grids["af_type"].fillna("non_ig")
    grids["af_ig_"]  = grids["af_ig_"].fillna(0)
    return grids


def create_result_with_nearest_points(grid_gdf, points_gdf):
    """Assign the nearest AF point (and its date/time) to each grid cell."""
    grid   = grid_gdf.copy()
    points = points_gdf.copy()
    points["ACQ_DATE"] = pd.to_datetime(points["ACQ_DATE"]).dt.date

    results = []
    for _, grid_row in grid.iterrows():
        centroid = grid_row.geometry.centroid
        points["distance"] = points.geometry.distance(centroid)
        closest = points.loc[points["distance"].idxmin()]
        row = grid_row.to_dict()
        row.update(
            nearest_point     = closest.geometry,
            n_ACQ_DATE        = closest["ACQ_DATE"],
            n_ACQ_TIME        = closest["ACQ_TIME"],
            distance_to_point = closest["distance"],
            nearest_point_lat = closest.geometry.y,
            nearest_point_lon = closest.geometry.x,
        )
        results.append(row)
    return gpd.GeoDataFrame(results, crs=grid.crs)


def process_fire(hull_shp, point_folder, grid_province, year):
    """Run grid creation for a single fire event. Returns (gdf, fire_id, year_name)."""
    hull_gdf = gpd.read_file(hull_shp)
    if hull_gdf.empty:
        return None

    fire_id   = hull_gdf["fire_id"].iloc[0]
    year_name = hull_gdf["year"].iloc[0] if "year" in hull_gdf.columns else year

    # Match corresponding point shapefile
    point_fname = os.path.basename(hull_shp).replace("_convex.shp", ".shp")
    point_path  = os.path.join(point_folder, point_fname)
    if not os.path.exists(point_path):
        return None
    point_gdf = gpd.read_file(point_path)
    if point_gdf.empty:
        return None

    gdf = create_grid_from_hull(hull_gdf, grid_province)
    if gdf is None or gdf.empty:
        return None

    gdf = prep_point(point_gdf, gdf)
    gdf = create_result_with_nearest_points(gdf, point_gdf)

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

        print("  Loading province grid ...")
        grid_shp = edit_grids(gpd.read_file(GRID_PATHS[province]))

        for year in YEAR_RANGE:
            hull_list = glob.glob(
                os.path.join(HULL_INPUT_PATHS[province], f"*fire_{year}_*.shp")
            )
            if not hull_list:
                continue
            print(f"\n  Year {year}: {len(hull_list)} fires")

            for hull_shp in tqdm(hull_list, desc=f"Year {year}", unit="fire", leave=False):
                try:
                    result = process_fire(hull_shp, POINT_INPUT_PATHS[province], grid_shp, year)
                    if result is not None:
                        gdf, fire_id, year_name = result
                        out_path = os.path.join(out_folder, f"fire_{year_name}_id_{fire_id}_grid.shp")
                        gdf.to_file(out_path)
                except Exception as e:
                    print(f"  Failed {os.path.basename(hull_shp)}: {e}")
                finally:
                    gc.collect()

    print("\n Creating grids completed.")
