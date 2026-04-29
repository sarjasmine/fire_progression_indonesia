"""
02_add_fire_point_attributes.py
=================================
Adds attributes to clustered AF point shapefiles: temporal duration,
burned area validation, coordinate conversion, and rate-of-spread (ROS).

Pipeline per fire shapefile:
    1. Add year, month, ACQ_DATETIME, province columns
    2. Filter events burning >= 2 days
    3. Load MODIS burned area and validate intersection within 15-day window
    4. Convert lon/lat degrees to projected metres (LON_M, LAT_M)
    5. Compute point-level and event-level rate-of-spread (ROS) via KDTree
    6. Save enriched shapefile
"""

import os
import gc
import glob

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import KDTree
from pyproj import Transformer
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths 
# ---------------------------------------------------------------------------

INPUT_PATHS = {
    "sumatra":    "data_preprocess/fire_event_shp_modis_af/fire_id_empty/v4/points_60/sumatra",
    "kalimantan": "data_preprocess/fire_event_shp_modis_af/fire_id_empty/v4/points_60/kalimantan",
}

FIRE_BA_PATHS = {
    "sumatra":    "data_preprocess/fire_event_shp_modis_ba/sumatra",
    "kalimantan": "data_preprocess/fire_event_shp_modis_ba/kalimantan",
}

OUTPUT_PATHS = {
    "sumatra":    "data_preprocess/fire_event_shp_modis_af/attributes/mapbiomas_c4_ver/v4_500m_60cl/1_fire_character/sumatra",
    "kalimantan": "data_preprocess/fire_event_shp_modis_af/attributes/mapbiomas_c4_ver/v4_500m_60cl/1_fire_character/kalimantan",
}

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

ALBERS_INDO_CRS = (
    "+proj=aea +lon_0=107.017 +lat_0=-0.13815 "
    "+lat_1=4.0417250000000005 +lat_2=-4.3180250000000004 "
    "+datum=WGS84 +units=m +no_defs"
)

DAY_DIFFERENCE = 15   # max days between AF and burned area date
YEAR_RANGE     = range(2001, 2025)

# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

def change_fire_attr(gdf_af, province_folder_path):
    """Add year, month, ACQ_DATETIME, and province columns."""
    gdf_af = gdf_af.copy()
    gdf_af["ACQ_DATE"] = pd.to_datetime(gdf_af["ACQ_DATE"])
    gdf_af["year"]  = gdf_af["ACQ_DATE"].dt.year
    gdf_af["month"] = gdf_af["ACQ_DATE"].dt.month

    acq_time_str = gdf_af["ACQ_TIME"].astype(str).str.zfill(4)
    gdf_af["ACQ_DATETIME"] = gdf_af["ACQ_DATE"] + acq_time_str.apply(
        lambda x: pd.Timedelta(hours=int(x[:2]), minutes=int(x[2:]))
    )

    return gdf_af


def add_temporal_attr(gdf, fire_id_column="fire_id"):
    """
    Filter fire events to those burning >= 2 days and add burn_duration column.
    Returns None if no valid events found.
    """
    gdf = gdf.copy()
    min_date = gdf.groupby(["year", fire_id_column])["ACQ_DATE"].transform("min")
    max_date = gdf.groupby(["year", fire_id_column])["ACQ_DATE"].transform("max")
    gdf["burn_duration"] = (max_date - min_date).dt.days + 1

    event_durations = gdf.groupby(["year", fire_id_column])["burn_duration"].first()
    valid_events    = event_durations[event_durations >= 2].index

    if len(valid_events) == 0:
        return None

    mask = gdf.set_index(["year", fire_id_column]).index.isin(valid_events)
    return gdf[mask].copy()


def add_burned_area(gdf, fire_ba_fp, crs=ALBERS_INDO_CRS):
    """
    Load and concat MODIS burned area shapefiles for all years present in gdf.
    Converts DOY burn_date to a proper date column (burn_date_ba).
    """
    ba_list = []
    for year in gdf["year"].unique():
        fp = os.path.join(fire_ba_fp, f"fire_ba_{year}.shp")
        gdf_ba = gpd.read_file(fp).to_crs(crs)
        gdf_ba["time"] = pd.to_datetime(gdf_ba["time"])
        gdf_ba["year"] = gdf_ba["time"].dt.year
        gdf_ba["burn_date_ba"] = (
            pd.to_datetime(gdf_ba["year"], format="%Y")
            + pd.to_timedelta(gdf_ba["burn_date"] - 1, unit="d")
        )
        ba_list.append(gdf_ba)
    return gpd.GeoDataFrame(pd.concat(ba_list, ignore_index=True))


def filter_burned_area(gdf_af, gdf_ba, day_difference=DAY_DIFFERENCE):
    """
    Keep only AF fire events whose points intersect a burned area polygon
    within the temporal window (0 to day_difference days after AF date).
    Adds min_date_ba column with the earliest confirmed burn date per fire.
    """
    result = []
    for year in gdf_af["year"].unique():
        subset = gdf_af[gdf_af["year"] == year].copy()

        gdf_intersect = gpd.sjoin(
            subset, gdf_ba[["burn_date_ba", "geometry"]],
            how="left", predicate="intersects"
        )
        gdf_intersect["date_diff"] = (
            gdf_intersect["burn_date_ba"] - gdf_intersect["ACQ_DATE"]
        ).dt.days

        filtered = gdf_intersect[
            (gdf_intersect["date_diff"] >= 0) &
            (gdf_intersect["date_diff"] <= day_difference)
        ].drop(columns=["date_diff"])

        unique_ids  = filtered["fire_id"].unique()
        gdf_filtered = subset[subset["fire_id"].isin(unique_ids)].copy()

        # Pick the most recent BA
        burn_dates = (
            filtered.groupby("fire_id")["burn_date_ba"].min()
            .reset_index()
            .rename(columns={"burn_date_ba": "min_date_ba"})
        )
        gdf_filtered = gdf_filtered.merge(burn_dates, on="fire_id", how="left")
        result.append(gdf_filtered)

    return gpd.GeoDataFrame(pd.concat(result, ignore_index=True))


def xy_degrees_to_meters(gdf, crs=ALBERS_INDO_CRS):
    """Add LON_M and LAT_M columns (projected metres) from LONGITUDE / LATITUDE."""
    gdf = gdf.copy()
    transformer = Transformer.from_crs("epsg:4326", crs, always_xy=True)
    gdf["LON_M"], gdf["LAT_M"] = transformer.transform(
        gdf["LONGITUDE"].values, gdf["LATITUDE"].values
    )
    return gdf


def add_ros(gdf):
    for year in gdf['year'].unique():
        # Get the wanted year
        subset_gdf = gdf[gdf['year'] == year].copy()
        # Sort by fire_id and time
        subset_gdf = subset_gdf.sort_values(['fire_id', 'ACQ_DATE']).copy()
        
        # Initiate lists
        ros_results = []
        dist_results = []
        days_diff_results = []
        total_dist_per_fire = {}
        step1_results = []
        step2_results = []
        cumul_days_results = []
        
        # Also store fire durations for each fire
        fire_durations_dict = {}
        
        # Loop
        for fire_id, fire_df in gdf.groupby('fire_id'):
            # Get the unique lat and lon from MODIS AF
            fire_df_unique = fire_df.drop_duplicates(subset=['LON_M', 'LAT_M', 'ACQ_DATE'])
            # Sort by date
            fire_df_unique = fire_df_unique.sort_values('ACQ_DATE')
            
            # Get unique dates in order
            unique_dates = fire_df_unique['ACQ_DATE'].unique()
            
            # Calculate cumulative days from start AND fire duration
            if len(unique_dates) > 0:
                start_date = unique_dates[0]
                end_date = unique_dates[-1]
                
                # Calculate fire duration (inclusive)
                fire_duration = (end_date - start_date).days + 1
                fire_durations_dict[fire_id] = fire_duration
                
                # Calculate cumulative days for each date
                cumul_days_from_start = {}
                for date in unique_dates:
                    days_since_start = (date - start_date).days
                    cumul_days_from_start[date] = days_since_start
            else:
                fire_durations_dict[fire_id] = 1
                cumul_days_from_start = {}
            
            # Group by day
            day_groups = list(fire_df_unique.groupby('ACQ_DATE'))
            
            # Initialize dictionaries
            ros_dict = {}
            dist_dict = {}
            days_diff_dict = {}
            step1_dict = {}
            step2_dict = {}
            cumul_days_dict = {}
            
            # First day initialization
            if len(day_groups) > 0:
                first_date = day_groups[0][0]
                first_day_df = day_groups[0][1]
                
                for row in first_day_df.itertuples():
                    key = (row.LON_M, row.LAT_M, row.ACQ_DATE)
                    ros_dict[key] = 0
                    dist_dict[key] = 0
                    days_diff_dict[key] = 0
                    step1_dict[key] = row.index_af
                    step2_dict[key] = row.index_af
                    cumul_days_dict[key] = cumul_days_from_start.get(first_date, 0)
                
                # Create mapping for previous day
                prev_coords_to_index = {}
                for row in first_day_df.itertuples():
                    prev_coords_to_index[(row.LON_M, row.LAT_M)] = row.index_af
                
                # Loop over consecutive day pairs
                for i in range(1, len(day_groups)):
                    prev_day, prev_points_df = day_groups[i-1]
                    curr_day, curr_points_df = day_groups[i]
                    
                    # Prepare coordinate arrays
                    prev_coords = []
                    prev_indices_list = []
                    for row in prev_points_df.itertuples():
                        prev_coords.append([row.LON_M, row.LAT_M])
                        prev_indices_list.append(row.index_af)
                    
                    curr_coords = []
                    curr_indices_map = {}
                    for row in curr_points_df.itertuples():
                        curr_coords.append([row.LON_M, row.LAT_M])
                        curr_indices_map[(row.LON_M, row.LAT_M)] = row.index_af
                    
                    # Calculate days difference
                    days_diff = (curr_day - prev_day).days
                    
                    if len(prev_coords) > 0 and len(curr_coords) > 0:
                        # Convert to numpy arrays
                        prev_coords_array = np.array(prev_coords)
                        curr_coords_array = np.array(curr_coords)
                        
                        # Find nearest neighbors
                        tree = KDTree(prev_coords_array)
                        dists, neighbor_indices = tree.query(curr_coords_array, k=1)
                        
                        # Calculate ROS values
                        ros_values = dists / days_diff if days_diff > 0 else np.zeros_like(dists)
                        
                        # Process each current point
                        for j in range(len(curr_coords)):
                            lon_m, lat_m = curr_coords[j]
                            curr_coord_key = (lon_m, lat_m)
                            curr_index_af = curr_indices_map.get(curr_coord_key)
                            
                            # Get previous point's index_af
                            prev_index_af = None
                            if neighbor_indices[j] < len(prev_indices_list):
                                prev_index_af = prev_indices_list[neighbor_indices[j]]
                            
                            key = (lon_m, lat_m, curr_day)
                            ros_dict[key] = ros_values[j]
                            dist_dict[key] = dists[j]
                            days_diff_dict[key] = days_diff
                            step1_dict[key] = prev_index_af if prev_index_af is not None else curr_index_af
                            step2_dict[key] = curr_index_af
                            cumul_days_dict[key] = cumul_days_from_start.get(curr_day, 0)
                    
                    # Update for next iteration
                    prev_coords_to_index = curr_indices_map.copy()
                
                # Calculate total spread distance
                total_dist = sum(dist_dict.values())
            else:
                total_dist = 0
            
            total_dist_per_fire[fire_id] = total_dist

        # Map results back to all rows of this fire
        for row in fire_df.itertuples():
            key = (row.LON_M, row.LAT_M, row.ACQ_DATE)
            ros_results.append(ros_dict.get(key, 0))
            dist_results.append(dist_dict.get(key, 0))
            days_diff_results.append(days_diff_dict.get(key, 0))
            cumul_days_results.append(cumul_days_dict.get(key, 0))
            step1_results.append(step1_dict.get(key, row.index_af))
            step2_results.append(step2_dict.get(key, row.index_af))

    gdf = gdf.copy()
    gdf["ros_point"]       = ros_results
    gdf["dist_point"]      = dist_results
    gdf["days_diff_point"] = days_diff_results
    gdf["dist_event"]      = gdf["fire_id"].map(total_dist_per_fire)
    gdf["cumm_days"]       = cumul_days_results
    gdf["fire_duration"]   = gdf["fire_id"].map(fire_durations_dict)

    gdf["ros_avg_cumulative"] = np.where(
        gdf["cumm_days"] > 0, gdf["dist_event"] / gdf["cumm_days"], 0
    )
    gdf["ros_avg_duration"] = np.where(
        gdf["fire_duration"] > 0, gdf["dist_event"] / gdf["fire_duration"], 0
    )
    gdf["step1_id"] = step1_results
    gdf["step2_id"] = step2_results
    return gdf


def process_fire(shp, province):
    """
    Run the full attribute enrichment pipeline for one fire shapefile.
    Returns enriched GeoDataFrame or None if event is filtered out.
    """
    gdf = gpd.read_file(shp)
    if gdf.empty:
        return None

    # Step 1: add year, month, datetime, province
    gdf = change_fire_attr(gdf, INPUT_PATHS[province])

    # Step 2: filter to >= 2 day burns
    gdf = add_temporal_attr(gdf)
    if gdf is None or gdf.empty:
        return None

    # Step 3: burned area validation
    gdf_ba     = add_burned_area(gdf, FIRE_BA_PATHS[province])
    gdf        = filter_burned_area(gdf, gdf_ba)
    if gdf.empty:
        return None

    # Step 4: convert coordinates to metres
    gdf = xy_degrees_to_meters(gdf)

    # Step 5: compute ROS
    gdf = add_ros(gdf)

    return gdf


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
                fire_gdf = fire_gdf_edited = None
                try:
                    fire_gdf_edited = process_fire(shp, province)
                    if fire_gdf_edited is not None and not fire_gdf_edited.empty:
                        year_name = fire_gdf_edited["ACQ_DATE"].dt.year.iloc[0]
                        fire_id   = fire_gdf_edited["fire_id"].iloc[0]
                        out_path  = os.path.join(out_folder, f"fire_{year_name}_id_{fire_id}.shp")
                        fire_gdf_edited.to_file(out_path)
                except Exception as e:
                    print(f"  Failed {os.path.basename(shp)}: {e}")
                finally:
                    del fire_gdf, fire_gdf_edited
                    gc.collect()

            gc.collect()

    print("\nAdd point characters (rate of spread) complete.")
