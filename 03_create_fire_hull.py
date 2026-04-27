"""
03_create_fire_hull.py
======================
Creates convex hull polygons for each fire event from the clustered AF point shapefiles.

Pipeline:
    1. Load per-fire AF point shapefiles
    2. Build an extended convex hull polygon per fire event
    3. Save one polygon shapefile per fire event per year
"""

import os
import gc
import glob

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import MultiPoint, Point, LineString
import alphashape
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

INPUT_PATHS = {
    "sumatra":    "data_preprocess/fire_event_shp_modis_af/attributes/mapbiomas_c4_ver/v4_500m_60cl/1_fire_character/sumatra",
    "kalimantan": "data_preprocess/fire_event_shp_modis_af/attributes/mapbiomas_c4_ver/v4_500m_60cl/1_fire_character/kalimantan",
}

OUTPUT_PATHS = {
    "sumatra":    "data_preprocess/fire_event_shp_modis_af/attributes/mapbiomas_c4_ver/v4_500m_60cl/2_create_alpha_hull/sumatra",
    "kalimantan": "data_preprocess/fire_event_shp_modis_af/attributes/mapbiomas_c4_ver/v4_500m_60cl/2_create_alpha_hull/kalimantan",
}

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

BUFFER_DISTANCE = 250   # metres applied to convex hull
YEAR_RANGE      = range(2001, 2025)

# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

def create_fire_polygon_convex(points_array, buffer_distance=BUFFER_DISTANCE):
    """
    Create a buffered convex hull polygon from an array of (x, y) coordinates.

    Parameters
    ----------
    points_array : array-like of shape (n, 2)
    buffer_distance : float, metres for the convex hull

    Returns
    -------
    Convex hull with buffer applied
    """
    hull = MultiPoint(points_array).convex_hull
    return hull.buffer(buffer_distance) if buffer_distance > 0 else hull


def build_hull_gdf(fire_gdf, fire_id, year, buffer_distance=BUFFER_DISTANCE):
    """
    Build a single-row GeoDataFrame containing the convex hull polygon for one fire.

    Parameters
    ----------
    fire_gdf : GeoDataFrame
        AF point data for one fire event.
    fire_id : int or str
    year : int
    buffer_distance : float

    Returns
    -------
    (polygon_gdf, point_count) or (None, 0) on failure.
    """
    try:
        points = np.array([(g.x, g.y) for g in fire_gdf.geometry])
        points_count = len(points)

        polygon = create_fire_polygon_convex(points, buffer_distance)

        dates = pd.to_datetime(fire_gdf["ACQ_DATE"]) if "ACQ_DATE" in fire_gdf.columns else None

        polygon_gdf = gpd.GeoDataFrame(
            {
                "fire_id":  [fire_id],
                "year":     [year],
                "af_count": [points_count],
                "area_km2": [polygon.area / 1_000_000],
                "buffer_m": [buffer_distance],
                "date_min": [dates.min() if dates is not None else None],
                "date_max": [dates.max() if dates is not None else None],
            },
            geometry=[polygon],
            crs=fire_gdf.crs,
        )
        return polygon_gdf, points_count

    except Exception as e:
        print(f"  Error building hull: {e}")
        return None, 0


def process_hull_year(year, province, input_folder, output_folder):
    """Build and save convex hull polygons for all fire events in one year."""
    shp_list = glob.glob(os.path.join(input_folder, f"*fire_{year}_*.shp"))
    if not shp_list:
        return

    print(f"  {year}: {len(shp_list)} fires")

    total_fires = 0
    for shp in tqdm(shp_list, desc=f"Year {year}", unit="fire", leave=False):
        fire_gdf = None
        try:
            fire_gdf = gpd.read_file(shp)
            if fire_gdf.empty:
                continue

            year_name = fire_gdf["ACQ_DATE"].dt.year.iloc[0] if "ACQ_DATE" in fire_gdf.columns else year
            fire_id   = fire_gdf["fire_id"].iloc[0] if "fire_id" in fire_gdf.columns else f"unknown_{total_fires}"

            polygon_gdf, _ = build_hull_gdf(fire_gdf, fire_id, year_name, BUFFER_DISTANCE)

            if polygon_gdf is not None and not polygon_gdf.empty:
                out_path = os.path.join(output_folder, f"fire_{year_name}_id_{fire_id}_convex.shp")
                polygon_gdf.to_file(out_path)
                total_fires += 1

        except Exception as e:
            print(f"  Failed: {e}")
        finally:
            del fire_gdf
            gc.collect()

    print(f"  Saved {total_fires} hull polygons for {year}.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    for province in ["sumatra", "kalimantan"]:
        out = OUTPUT_PATHS[province]
        os.makedirs(out, exist_ok=True)
        print(f"\n{'='*50}")
        print(f"Province: {province.upper()}")
        print(f"{'='*50}")
        for year in YEAR_RANGE:
            process_hull_year(year, province, INPUT_PATHS[province], out)
            gc.collect()

    print("\nCreating convex hull complete.")
