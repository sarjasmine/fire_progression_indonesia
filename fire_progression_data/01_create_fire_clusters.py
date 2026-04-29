"""
01_create_fire_clusters.py
==========================
Creates clustered active fire (AF) point shapefiles from raw MODIS FIRMS data.

Pipeline:
    1. Load MODIS AF shapefiles for Sumatra and Kalimantan
    2. Reproject to custom Albers Equal Area CRS
    3. Filter by confidence level
    4. Spatial join with province grids
    5. Breadth-First Search (BFS) clustering by proximity + temporal window
    6. Save one shapefile per fire event per year
"""

import os
import gc
import glob
from collections import deque

import numpy as np
import pandas as pd
import geopandas as gpd
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Modified AF from NASA FIRMS, intersected with regions
MODIS_AF_PATHS = {
    "sumatra":    "MODIS_AF/Sumatra/sumatra_firms_modis.shp",
    "kalimantan": "MODIS_AF/Kalimantan/kalimantan_firms_modis.shp",
}

# Modified 500-m grid
GRID_PATHS = {
    "sumatra":    "data_preprocess/grids/sumatra_grids_ba.shp",
    "kalimantan": "data_preprocess/grids/kalimantan_grids_ba.shp",
}

OUTPUT_PATHS = {
    "sumatra":    "data_preprocess/fire_event_shp_modis_af/fire_id_empty/v4/points_60/sumatra",
    "kalimantan": "data_preprocess/fire_event_shp_modis_af/fire_id_empty/v4/points_60/kalimantan",
}

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

# Customized CRS
ALBERS_INDO_CRS = (
    "+proj=aea +lon_0=107.017 +lat_0=-0.13815 "
    "+lat_1=4.0417250000000005 +lat_2=-4.3180250000000004 "
    "+datum=WGS84 +units=m +no_defs"
)

# Determined variables
CONFIDENCE_LEVEL = 60
BUFFER_DISTANCE  = 500   # metres, spatial neighbourhood for clustering
DAY_DIFFERENCE   = 4     # days, temporal window for clustering
MIN_MEMBERS      = 3     # minimum AF points per cluster
YEAR_RANGE       = range(2001, 2025) # From 2001-2024

# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def edit_af(shp, confidence_level=60):
    """Filter AF points by confidence threshold and keep relevant columns."""
    conf_filter = shp.loc[shp["CONFIDENCE"] >= confidence_level]
    return conf_filter[[
        "LATITUDE", "LONGITUDE", "BRIGHTNESS", "ACQ_DATE", "ACQ_TIME",
        "CONFIDENCE", "BRIGHT_T31", "FRP", "geometry"
    ]]


def load_af_for_year(year, province, crs=ALBERS_INDO_CRS):
    """
    Load, reproject, filter, and spatially join AF points to the province grid.

    Returns a GeoDataFrame of AF points with grid_id assigned.
    """
    gdf = gpd.read_file(MODIS_AF_PATHS[province]).to_crs(crs)
    # Extract the year
    gdf["ACQ_DATE"] = pd.to_datetime(gdf["ACQ_DATE"])
    gdf = gdf[gdf["ACQ_DATE"].dt.year == year]

    # Add region label so each point knows which province it belongs to
    gdf["region"] = province.capitalize()

    # Filter the AF
    gdf = edit_af(gdf, confidence_level=CONFIDENCE_LEVEL)
    gdf = gdf.reset_index(drop=False).rename(columns={"index": "index_af"})

    grid = gpd.read_file(GRID_PATHS[province]).to_crs(crs)
    grid = grid.reset_index().rename(columns={"index": "grid_id"})

    joined = gpd.sjoin(gdf, grid, how="left", predicate="within")
    joined = joined.sort_values("grid_id").drop_duplicates(subset="index_af")
    return joined


def create_fire_cluster(gdf_points, buffer_distance=BUFFER_DISTANCE,
                        day_difference=DAY_DIFFERENCE, min_members=MIN_MEMBERS,
                        crs=ALBERS_INDO_CRS):
    """
    Cluster AF points using BFS with spatial buffer + temporal window.

    Parameters
    ----------
    gdf_points : GeoDataFrame
        Must have columns: index_af, ACQ_DATE, geometry.
    buffer_distance : float
        Spatial buffer in metres.
    day_difference : int
        Max days between points to belong to the same cluster.
    min_members : int
        Minimum cluster size retained.

    Returns
    -------
    GeoDataFrame with fire_id and af_type columns added.
    """
    gdf = gdf_points.copy().to_crs(crs)
    gdf["fire_id"] = -1
    gdf["af_type"] = "ig"  # ignition type from MODIS AF
    gdf["ACQ_DATE"] = pd.to_datetime(gdf["ACQ_DATE"])

    sindex = gdf.sindex
    af_fires_id = gdf["index_af"].unique()
    clustered_ids = set()
    current_cluster_id = 0

    for af_fire in af_fires_id:
        if af_fire in clustered_ids:
            continue

        queue = deque([af_fire])
        cluster_members = set()

        while queue:
            current_id = queue.popleft()
            if current_id in cluster_members:
                continue
            cluster_members.add(current_id)

            current_fire = gdf[gdf["index_af"] == current_id]
            current_geom = current_fire.geometry.unary_union
            current_dates = current_fire["ACQ_DATE"]

            buffered = current_geom.buffer(buffer_distance)
            possible_idx = list(sindex.intersection(buffered.bounds))
            possible_matches = gdf.iloc[possible_idx]

            for _, neighbor in possible_matches.iterrows():
                nid = neighbor["index_af"]
                if nid in clustered_ids or nid in cluster_members:
                    continue
                date_match = any(
                    abs(d - neighbor["ACQ_DATE"]) <= pd.Timedelta(days=day_difference)
                    for d in current_dates
                )
                if date_match:
                    queue.append(nid)

        gdf.loc[gdf["index_af"].isin(cluster_members), "fire_id"] = current_cluster_id
        clustered_ids.update(cluster_members)
        current_cluster_id += 1

    # Drop clusters smaller than min_members
    counts = gdf.groupby("fire_id").size()
    valid_ids = counts[counts >= min_members].index
    gdf = gdf[gdf["fire_id"].isin(valid_ids)]
    return gpd.GeoDataFrame(gdf, geometry="geometry", crs=gdf.crs)


def process_fire_clusters_year(year, province, output_folder):
    """Cluster and export AF point shapefiles for one year and province."""
    print(f"  Processing {year} ...")
    try:
        gdf = load_af_for_year(year, province)
        clusters = create_fire_cluster(gdf)
        for fid, fire_gdf in clusters.groupby("fire_id"):
            out_path = os.path.join(output_folder, f"fire_{year}_id_{fid}.shp")
            fire_gdf.to_file(out_path)
    except Exception as e:
        print(f"  Failed for {year}: {e}")


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
            process_fire_clusters_year(year, province, out)
            gc.collect()

    print("\nFire clusters complete.")
