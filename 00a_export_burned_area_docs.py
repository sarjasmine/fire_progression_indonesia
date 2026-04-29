"""
0_export_burned_area.py
========================
Exports MODIS MCD64A1 burned area data to shapefiles for Sumatra and Kalimantan.

Pipeline:
    1. Download MODIS MCD64A1 as NetCDF at: https://appeears.earthdatacloud.nasa.gov/
    2. Open MODIS MCD64A1 NetCDF and clip to province boundary, download the Sumatra and Kalimantan: https://tanahair.indonesia.go.id/portal-web/
    3. Create a 500m meshgrid shapefile for Sumatra and Kalimantan
    4. Extract valid burned pixels (burn_date > 0) per year as point GeoDataFrame
    5. Spatial join burned points with province grid
    6. Save one burned area shapefile per province per year as: fire_ba_{year}.shp

Inputs:
    - MODIS MCD64A1 NetCDF     : MODIS_BA/MCD64A1.061_{Sumatra,Borneo}/MCD64A1.061_500m_aid0001_{year}.nc # Borneo = Main Island name for Kalimantan (Indonesia part)
    - Province boundaries      : INA_GEOPORTAL/ADMIN_edited/{Sumatra,Kalimantan}_island_dissolved.shp

Outputs:
    - Province grids (once)    : data_preprocess/{sumatra,kalimantan}_grids_ba.shp
    - Burned area per year     : data_preprocess/fire_event_shp_modis_ba/{sumatra,kalimantan}/fire_ba_{year}.shp

Key variables per output row:
    burn_date : int   Day-of-year (DOY) of first detected burn (1–366)
    time      : str   Monthly timestamp of the MODIS granule
    grid_id   : int   Grid cell ID linking to the province meshgrid
"""





# Folder path
modi
# ---------------------------------------------------------------------------
# Library
# ---------------------------------------------------------------------------

import os
import geopandas as gpd
import glob

from shapely.geometry import box, Point

import rasterio              
import rasterio.features     
import rasterio.warp         
from scipy.ndimage import label, generate_binary_structure

import xarray
import rioxarray             


# ---------------------------------------------------------------------------
# Function reference
# ---------------------------------------------------------------------------
# Open the MODIS shapefiles

# File path
modis_folder = 'MODIS_BA'
modis_sumatra_folder = os.path.join(modis_folder, 'MCD64A1.061_Sumatra')
modis_borneo_folder = os.path.join(modis_folder, 'MCD64A1.061_Borneo')


# Folder to admin shapefile
shp_folder = 'INA_GEOPORTAL/ADMIN_edited'
sum_shp_fp = os.path.join(shp_folder, 'Sumatra_island_dissolved.shp')
sum_shp = gpd.read_file(sum_shp_fp)
kal_shp_fp = os.path.join(shp_folder, 'Kalimantan_island_dissolved.shp')
kal_shp = gpd.read_file(kal_shp_fp)


def open_set_ds(ds_fp, year, shp_boundary):
    """
    Open a MODIS MCD64A1 NetCDF, set CRS to EPSG:4326, and clip to province boundary.

    Parameters
    ----------
    ds_fp : str
        Folder containing MCD64A1.061_500m_aid0001_{year}.nc files.
    year : int
    shp_boundary : GeoDataFrame
        Province boundary polygon used for clipping.

    Returns
    -------
    xarray.Dataset clipped to province extent.
    """
    ds_filepath = os.path.join(ds_fp, f"MCD64A1.061_500m_aid0001_{year}.nc")
    ds = xr.open_dataset(ds_filepath)
    # Set the crs
    ds = ds.rio.write_crs("EPSG:4326")
    # Crop to the shp boundary
    ds = ds.rio.clip(shp_boundary.geometry.values, shp_boundary.crs, drop = True)
    return ds

    ...


def create_meshgrid_province(ds):
    """
    Create a 500m grid of box polygons matching the MODIS pixel layout.
    Run ONCE per province — output is reused across all years.

    Each grid cell is a box polygon centred on the MODIS pixel lat/lon,
    with width/height equal to the dataset resolution (~0.00449 degrees ≈ 500m).

    Parameters
    ----------
    ds : xarray.Dataset
        Opened MODIS dataset with lat/lon coordinates.

    Returns
    -------
    GeoDataFrame with columns: lon, lat, geometry (box polygons), CRS=EPSG:4326.

    Outputs
    -------
    Saved to:
        data_preprocess/sumatra_grids_ba.shp
        data_preprocess/kalimantan_grids_ba.shp
    """
    lat = ds['lat'].values
    lon = ds['lon'].values   
    # Get degree resolutin
    res_lat = float(lat[1] - lat[0])
    res_lon = float(lon[1] - lon[0])
    # Create the meshgrid
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    # Flatten the arrays
    flat_lat = lat_grid.flatten()
    flat_lon = lon_grid.flatten()
    geometries = [box(lon - res_lon / 2,
                      lat - res_lat / 2,
                      lon + res_lon / 2,
                      lat + res_lat / 2)
                  for lon, lat in zip(flat_lon, flat_lat)]
    # Create gdf
    gdf = gpd.GeoDataFrame({
        'lon' : flat_lon,
        'lat' : flat_lat,
        'geometry' : geometries,
    }, crs='epsg:4326')
    return gdf
    ...


def extract_burned_points_to_gdf(ds):
    """
    Extract all valid burned pixels from a MODIS dataset as a point GeoDataFrame.

    A pixel is valid if Burn_Date > 0 (excludes unburned, water, non-processed).

    Parameters
    ----------
    ds : xarray.Dataset
        Clipped MODIS MCD64A1 dataset for one province and year.

    Returns
    -------
    GeoDataFrame with columns:
        burn_date : int   Day-of-year of detected burn
        time      : str   MODIS monthly timestamp
        geometry  : Point Pixel centre coordinate (EPSG:4326)
    """
    lat = ds["lat"].values
    lon = ds["lon"].values
    time = ds["time"].values

    # Create 2D lat/lon meshgrid
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    # Flatten lat/lon grid to match pixel layout
    flat_lat = lat_grid.flatten()
    flat_lon = lon_grid.flatten()
    

    records = []

    for t_index, t in enumerate(time):
        burn_date = ds["Burn_Date"].isel(time=t_index).values
        qa = ds["QA"].isel(time=t_index).values
        flat_burn = burn_date.flatten()
        flat_qa = qa.flatten()

        # Valid burn pixels (exclude NaNs, and optionally filter by 1–366 if DOY)
        valid_mask = (flat_burn > 0) & (~np.isnan(flat_burn))

        for b, q, latc, lonc in zip(
            flat_burn[valid_mask],
            flat_qa[valid_mask],
            flat_lat[valid_mask],
            flat_lon[valid_mask]
        ):
            records.append({
                "burn_date": b,
                "time": str(t),
                "geometry": Point(lonc, latc)
            })

    gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")
    return gdf
    ...

"""   
# Save the full grid
file_path = 'data_preprocess'
filename = os.path.join(file_path, 'sumatra_grids_ba.shp')
# Export as shp
full_grid.to_file(filename)
"""

# Open the full grid
sum_grid_shp_fp = 'data_preprocess/sumatra_grids_ba.shp'

sumatra_grid = gpd.read_file(sum_grid_shp_fp)

kal_grid_shp_fp = 'data_preprocess/kalimantan_grids_ba.shp'

kalimantan_grid = gpd.read_file(kal_grid_shp_fp)

# Output_folder
fire_output_sumatra = 'data_preprocess/fire_event_shp_modis_ba/sumatra'
fire_output_kalimantan = 'data_preprocess/fire_event_shp_modis_ba/kalimantan'


def process_fire_clusters(year, modis_folder, admin_shp, grid, output_folder):
    """
    Full pipeline for one province and one year:
        1. Open and clip MODIS dataset
        2. Extract valid burned points
        3. Spatial join with province grid
        4. Save to fire_ba_{year}.shp

    Parameters
    ----------
    year : int
    modis_folder : str
        Path to MCD64A1 NetCDF folder for the province.
    admin_shp : GeoDataFrame
        Province boundary for clipping.
    grid : GeoDataFrame
        Province meshgrid (output of create_meshgrid_province).
    output_folder : str
        Destination folder for fire_ba_{year}.shp.

    Output columns:
        burn_date : int   DOY of first burn detection
        time      : str   MODIS granule timestamp
        burn_id   : int   Index from valid_burn point GeoDataFrame
        grid_id   : int   Matched province grid cell ID
        geometry  : Polygon  Grid cell polygon
    """
    print(f"Processing year {year}...")
        try:
         # Load dataset and extract valid burned points
            ds = open_set_ds(modis_folder, year, admin_shp)
            valid_burn = extract_burned_points_to_gdf(ds)

        # Spatial join burned points with grid
            burned_area_gdf = gpd.sjoin(grid, valid_burn, how='inner', predicate='intersects')

            # Rename index_right to burn_id, reset index to grid_id
            burned_area_gdf = burned_area_gdf.rename(columns={'index_right': 'burn_id'})
            burned_area_gdf = burned_area_gdf.reset_index().rename(columns={'index': 'grid_id'})          
            out_path = os.path.join(output_folder, f"fire_ba_{year}.shp")
            burned_area_gdf.to_file(out_path)
            print(f"Saved: {out_path}")
        except Exception as e:
            print(f"Failed for {year}: {e}") 
    ...


# ---------------------------------------------------------------------------
# Run order 
# ---------------------------------------------------------------------------
for year in range(2023, 2026):
    print(f"Processing year {year}...")
    process_fire_clusters(year, modis_sumatra_folder, sum_shp, sumatra_grid, fire_output_sumatra)
    
    # Force garbage collection
    gc.collect()

for year in range(2023, 2026):
    print(f"Processing year {year}...")
    process_fire_clusters(year, modis_borneo_folder, kal_shp, kalimantan_grid, fire_output_kalimantan)
    # Force garbage collection
    gc.collect()