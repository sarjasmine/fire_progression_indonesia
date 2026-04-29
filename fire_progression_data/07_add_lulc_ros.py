"""
07_add_lulc_ros.py
==================
Adds MapBiomas LULC attributes and grid-level rate-of-spread (ROS) to fire grids.


Pipeline:
    1. Load fire grid shapefiles
    2. Merge MapBiomas LULC codes by grid_id
    3. Compute daily majority LULC, fire-level majority LULC, and burned area stats
    4. Simplify LULC sequence to 3-point and 5-point summaries
    5. Compute burn_duration and ros_km2_day from date_min / date_max
    6. Save enriched grid shapefile
"""

import os
import gc
import glob
from collections import Counter

import numpy as np
import pandas as pd
import geopandas as gpd
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths — edit these to match your local data layout
# ---------------------------------------------------------------------------

MAPBIOMAS_FOLDER = "MapBiomas_LULC/grid_lulc_removed_nan_lulc"

INPUT_PATHS = {
    "sumatra":    "data_preprocess/fire_event_shp_modis_af/attributes/mapbiomas_c4_ver/v4_500m_60cl/5_grids_climate/sumatra",
    "kalimantan": "data_preprocess/fire_event_shp_modis_af/attributes/mapbiomas_c4_ver/v4_500m_60cl/5_grids_climate/kalimantan",
}

OUTPUT_PATHS = {
    "sumatra":    "data_preprocess/fire_event_shp_modis_af/attributes/mapbiomas_c4_ver/v4_500m_60cl/6_grids_lulc_ros/sumatra",
    "kalimantan": "data_preprocess/fire_event_shp_modis_af/attributes/mapbiomas_c4_ver/v4_500m_60cl/6_grids_lulc_ros/kalimantan",
}

YEAR_RANGE = range(2001, 2025)

ALBERS_INDO_CRS = (
    "+proj=aea +lon_0=107.017 +lat_0=-0.13815 "
    "+lat_1=4.0417250000000005 +lat_2=-4.3180250000000004 "
    "+datum=WGS84 +units=m +no_defs"
)

# ---------------------------------------------------------------------------
# LULC reference (MapBiomas Indonesia Collection 4)
# ---------------------------------------------------------------------------

LULC_INDEX = {
    0: "No Data", 1: "Forest", 3: "Forest Formation", 5: "Mangrove",
    9: "Pulpwood Plantation", 10: "Non-Forest Natural Formation",
    13: "Non Forest Natural Vegetation", 18: "Agriculture",
    21: "Other Agriculture", 22: "Non Vegetated Area", 24: "Urban Area",
    25: "Other Non-Vegetation", 26: "Water Body", 27: "Non Observed",
    30: "Mining Pit", 31: "Aquaculture", 33: "River, Lake, Ocean",
    35: "Oil Palm", 40: "Rice Paddy", 76: "Peat Swamp Forest",
}

# ---------------------------------------------------------------------------
# LULC functions
# ---------------------------------------------------------------------------

def add_lulc(gdf, gdf_lulc, crs=ALBERS_INDO_CRS):
    """Merge MapBiomas LULC codes into fire grid by grid_id."""
    gdf_lulc_edit = gdf_lulc.to_crs(crs)
    gdf = gdf.merge(gdf_lulc_edit[["grid_id", "mapbiomas"]], on="grid_id", how="left")
    unique_lulc = gdf.groupby("fire_id")["mapbiomas"].unique().apply(list).to_dict()
    gdf["lulc_code"] = gdf["fire_id"].map(unique_lulc)
    return gdf


def add_major_lulc_area(gdf, lulc_col="mapbiomas", date_col="n_ACQ_DATE"):
    """
    Compute daily majority LULC, areas, and overall fire majority LULC.

    Adds columns: major_daily_lulc, major_daily_km2, major_daily_pct,
                  major_lulc, major_km2, major_pct, total_km2.
    Returns (gdf, lulc_seq)
    """
    pixel_area_km2      = (500 * 500) / 1e6 # MODIS pixel size = 500 meters
    total_fire_area_km2 = len(gdf) * pixel_area_km2

    # Determine majority LULC per day
    daily_major = (
        gdf.groupby(date_col)[lulc_col]
        .agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0])
        .reset_index(name="major_daily_lulc")
    )
    # Sequence of daily majority LULC
    lulc_seq = daily_major.sort_values(date_col)["major_daily_lulc"].tolist()

    # Merge back to tag pixels
    merged = gdf.merge(daily_major, on=date_col, how="left")
    merged["is_major"] = merged[lulc_col] == merged["major_daily_lulc"]

    # Count pixels that match majority and total pixels per day
    daily_counts = (
        merged.groupby(date_col)
        .agg(major_pixel_count=("is_major", "sum"), total_pixel_count=(lulc_col, "size"))
        .reset_index()
    )

    # Area calculations
    daily_counts["major_daily_km2"] = daily_counts["major_pixel_count"] * pixel_area_km2
    daily_counts["total_daily_km2"] = daily_counts["total_pixel_count"] * pixel_area_km2
    daily_counts["major_daily_pct"] = (
        daily_counts["major_daily_km2"] / daily_counts["total_daily_km2"]
    )

    # Merge daily results back into gdf
    gdf = gdf.merge(daily_major, on=date_col, how="left")
    gdf = gdf.merge(
        daily_counts[[date_col, "major_daily_km2", "major_daily_pct"]],
        on=date_col, how="left"
    )

    # Overall fire majority LULC
    overall_lulc    = gdf[lulc_col].mode().iloc[0]
    total_major_km2 = (gdf[lulc_col] == overall_lulc).sum() * pixel_area_km2

    gdf["major_lulc"] = overall_lulc
    gdf["major_km2"]  = total_major_km2
    gdf["major_pct"]  = total_major_km2 / total_fire_area_km2
    gdf["total_km2"]  = total_fire_area_km2
    return gdf, lulc_seq

# Remove only consecutive duplicates from a sequence. To summarize next-day same lulc
def remove_consecutive_duplicates(seq):
    if not seq:
        return []
    result = [seq[0]]
    for val in seq[1:]:
        if val != result[-1]:
            result.append(val)
    return result

def simplify_major_lulc_path_max3(seq):

    # Simplify daily_major_lulc sequence to maximum 3 points:
    # always keep first and last day major LULC
    # include the middle element as the mode of middle values (if different from first/last)
    
    if not seq:
        return []

    # Remove consecutive duplicates
    seq = remove_consecutive_duplicates(seq)

    # Start with first element
    simplified = [seq[0]]

    # Middle elements
    middle = seq[1:-1]
    
    # Keep only values different from first and last
    middle_filtered = [v for v in middle if v != seq[0] and v != seq[-1]]
    if middle_filtered:
        # pick the mode
        middle_mode = Counter(middle_filtered).most_common(1)[0][0]
        simplified.append(middle_mode)

    # Always include last element
    simplified.append(seq[-1])

    return simplified

def simplify_major_lulc_path_max5(seq):

    # Simplify LULC sequence to maximum 5 points:
    # always keep first and last elements
    # middle 3 points are the mode of 3 evenly spaced segments

    # Remove consecutive duplicates
    seq = remove_consecutive_duplicates(seq)

    if not seq:
        return []

    n = len(seq)
    if n <= 5:
        return seq  # already <=5 points

    simplified = [seq[0]]  # first element

    # Middle elements (exclude first and last)
    middle = seq[1:-1]
    m = len(middle)

    # Split middle into 3 segments
    if m < 3:
        # fewer than 3 middle elements → just use them all
        middle_points = middle
    else:
        # define segment boundaries
        indices = np.linspace(0, m, 4, dtype=int)  # 0, seg1_end, seg2_end, m
        middle_points = []
        for i in range(3):
            segment = middle[indices[i]:indices[i+1]]
            if segment:
                mode_value = Counter(segment).most_common(1)[0][0]
                middle_points.append(mode_value)

    simplified.extend(middle_points)
    simplified.append(seq[-1])  # last element

    return simplified

def add_simplified_path_columns(
    fire_gdf,
    fire_id_col='fire_id',
    acq_col='n_ACQ_DATE',
    major_daily_col='major_daily_lulc'
):
    gdf = fire_gdf.copy()
    gdf[acq_col] = pd.to_datetime(gdf[acq_col])

    path_records = []

    for fire_id, fire_df in gdf.groupby(fire_id_col):
        # One row per day (daily majority)
        daily_df = (
            fire_df[[acq_col, major_daily_col]]
            .drop_duplicates()
            .sort_values(acq_col)
        )

        if daily_df.empty:
            continue

        seq = daily_df[major_daily_col].tolist()

        first_lulc = seq[0]
        last_lulc  = seq[-1]

        # Numeric first→last code
        try:
            pair_code = int(first_lulc * 100 + last_lulc)
        except:
            pair_code = -1  # fallback for missing/invalid values

        path_records.append({
            fire_id_col: fire_id,
            's_seq_3': simplify_major_lulc_path_max3(seq),
            's_seq_5': simplify_major_lulc_path_max5(seq),
            'first_lulc': first_lulc,
            'last_lulc': last_lulc,
            'pair_end': pair_code
        })

    path_df = pd.DataFrame(path_records)

    return gdf.merge(path_df, on=fire_id_col, how='left')



# ---------------------------------------------------------------------------
# ROS function
# ---------------------------------------------------------------------------

def add_ros_grids(gdf):
    """Compute burn_duration and ros_km2_day from date_min / date_max columns."""
    gdf = gdf.copy()
    gdf["date_min"]      = pd.to_datetime(gdf["date_min"])
    gdf["date_max"]      = pd.to_datetime(gdf["date_max"])
    gdf["burn_duration"] = (gdf["date_max"] - gdf["date_min"]).dt.days + 1
    gdf["ros_km2_day"]   = gdf["total_km2"] / gdf["burn_duration"]
    return gdf


# ---------------------------------------------------------------------------
# Per-fire processor
# ---------------------------------------------------------------------------

def process_fire(shp, lulc_gdf, year):
    """Add LULC attributes and ROS to one fire grid shapefile."""
    gdf = gpd.read_file(shp)
    if gdf.empty:
        return None

    gdf["n_ACQ_DATE"] = pd.to_datetime(gdf["n_ACQ_DATE"], errors="coerce")

    gdf = add_lulc(gdf, lulc_gdf)
    gdf, _ = add_major_lulc_area(gdf, lulc_col="mapbiomas", date_col="n_ACQ_DATE")
    gdf = add_simplified_path_columns(gdf, acq_col="n_ACQ_DATE", major_daily_col="major_daily_lulc")
    gdf = add_ros_grids(gdf)

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

        province_label = province.capitalize()

        for year in YEAR_RANGE:
            shp_list = glob.glob(os.path.join(INPUT_PATHS[province], f"*fire_{year}_*.shp"))
            if not shp_list:
                continue

            lulc_fp = os.path.join(MAPBIOMAS_FOLDER, f"mapbiomas_{province_label}_{year}.shp")
            if not os.path.exists(lulc_fp):
                print(f"  LULC missing for {year}, skipping.")
                continue
            lulc_gdf = gpd.read_file(lulc_fp)

            print(f"\n  Year {year}: {len(shp_list)} fires")

            for shp in tqdm(shp_list, desc=f"Year {year}", unit="fire", leave=False):
                try:
                    result = process_fire(shp, lulc_gdf, year)
                    if result is not None:
                        gdf, fire_id, year_name = result
                        out_path = os.path.join(out_folder, f"fire_{year_name}_id_{fire_id}_grid.shp")
                        gdf.to_file(out_path)
                except Exception as e:
                    print(f"  Failed {os.path.basename(shp)}: {e}")
                finally:
                    gc.collect()

            del lulc_gdf
            gc.collect()

    print("\n Add LULC and ROS by Grids complete.")
