# Fire progression dataset, Indonesia
Fire progression in Sumatra and Kalimantan, Indonesia, 2001-2024. This repository contains two sets of fire dataset: 1) Fire cluster points, and 2) Fire grids. The fire data is based on MODIS Burned Area MCD64A1 and Active Fire MYD & MOD C6.1. The grids contain: NDVI (MODIS), EVI (MODIS), DEM (NASEM), LULC (Mapbiomas), and Climate (ERA5) variables.

These are sample codes (can be modified or adjusted based on needs) to produce the fire progression and the environmental variables.
The steps in order:
  1) 01_create_fire_clusters.py
  2) 02_add_fire_point_attributes.py
  3) 03_create_fire_hull.py
  4) 04_create_grids.py
  5) 05_add_ndvi_evi_dem.py
  6) 06_add_climate.py
  7) 07_add_lulc_ros.py

Structure folders:

└── fire_progression_data
    ├── 01_create_fire_clusters.py
    ├── 02_add_fire_point_attributes.py   
    ├── 03_create_fire_hull.py   
    ├── 04_create_grids.py   
    ├── 05_add_ndvi_evi_dem.py
    ├── 06_add_climate.py
    ├── 07_add_lulc_ros.py
└── preprocessing_data
    ├── 00a_export_burned_area_docs.py
    ├── 00b_preprocess_lulc.py
    ├── 00c_example_fire_visualization.ipynb
└── fire_sample_polygon  
    └── fire_2015_id_16860.cpg
    └── fire_2015_id_16860.dbf
    └── fire_2015_id_16860.prj
    └── fire_2015_id_16860.shp
    └── fire_2015_id_16860.shx
    └── fire_2015_id_16860_grid.cpg
    └── fire_2015_id_16860_grid.dbf
    └── fire_2015_id_16860_grid.prj
    └── fire_2015_id_16860_grid.shp
    └── fire_2015_id_16860_grid.shx


Fire points shapefile: a collection of clustered active fire hotspots in one event.

Fire grids shapefile: a grid-based fire polygon with 500 meter spatial resolution and daily timestamps.



Metadata:

Coordinate reference system:

ALBERS_INDO_CRS = (
    "+proj=aea +lon_0=107.017 +lat_0=-0.13815 "
    "+lat_1=4.0417250000000005 +lat_2=-4.3180250000000004 "
    "+datum=WGS84 +units=m +no_defs"

)



Data source:

1) MODIS Active Fire

MOD14&MYD14 C6.1, 1 km spatial resolution, near real time daily, confidence level >=60

https://firms.modaps.eosdis.nasa.gov/



2) MODIS Burned Area

MCD64A1, 500 m spatial resolution, monthly burned area

https://appeears.earthdatacloud.nasa.gov/



3) MODIS NDVI and EVI

MOD13A1.061, 500 m spatial resolution, 16-day temporal resolution

https://appeears.earthdatacloud.nasa.gov/



4) Mapbiomas LULC

Collection 4, 30 m spatial resolution, annual temporal resolution

https://indonesia.mapbiomas.org/en



5) ERA5 Climate

Single level hourly data, 0.25 degree spatial resolution

Variables: 2m dewpoint temperature (°C), 2m temperature (°C), total precipitation (m), convective precipitation (m), convective rain rate (mm/hour), evaporation (m), potential evaporation (m), instantaneous 10m wind gust(m/s), surface pressure (Pa), and mean sea level pressure (Pa), wind speed (m/s), and wind direction (degrees). Additonal derived variables: vapor pressure deficit (kPa) and relative humidity (%).

https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=download



6) NASADEM Elevation

Static, 30 meter spatial resolution

https://appeears.earthdatacloud.nasa.gov/


The outpus of these codes are fire progression in point clusters and grids shapefiles, available at Zenodo:
https://zenodo.org/records/19809792?preview=1&token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImI2ODkzMmNhLTQ4M2ItNDkwZi05YzQ5LTZiMjcyYWVmZDRkOSIsImRhdGEiOnt9LCJyYW5kb20iOiJiYmM3ZWI3YmQ4NWE1ODZhNWM0N2ZlNmIxYzJhNjk3MiJ9.D9HTqrbISYy3PcVVI9vPjJVmmTxfBgbqb645Xleij19DCpIeUUOU2-QFcaaA-YNz6alxpwoQvBbCaC5Kw0PuQg

The detailed method is available at: [paper link].



Contributors: Sarah Riadi & Manzhu Yu (Pennsylvania State University)
If you find any addional bugs and questions please send us a note!
