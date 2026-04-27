# Fire progression dataset, Indonesia
Fire progression in Sumatra and Kalimantan, Indonesia, 2001-2024. This repository contains two sets of fire dataset: 1) Fire cluster points, and 2) Fire grids. The fire data is based on MODIS Burned Area MCD64A1 and Active Fire MYD&amp;MOD C6.1. The grids contain: NDVI (MODIS), EVI (MODIS), DEM (NASEM), LULC (Mapbiomas), and Climate (ERA5) variables.

These are sample codes (can be modified or adjusted based on needs) to produce the fire progression and the environmental variables.
The steps in order:
  1) 01_create_fire_clusters.py
  2) 02_add_fire_point_attributes.py
  3) 03_create_fire_hull.py
  4) 04_create_grids.py
  5) 05_add_ndvi_evi_dem.py
  6) 06_add_climate.py
  7) 07_add_lulc_ros.py

The outpus of these codes are fire progression in point clusters and grids shapefiles, available at: [Zenodo link].

The detailed method is available at: [paper link].

If you find any addional bugs and questions please send us a note!
