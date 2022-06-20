# Introduction

## Publication

This notebook documents the workflow of the manuscript "Citizen science plant observations encode global trait patterns" by Sophie Wolf, Miguel D. Mahecha, Francesco Maria
Sabatini, Christian Wirth, Helge Bruelheide, Jens
Kattge, Álvaro Moreno Martínez, Karin Mora and Teja
Kattenborn.

## Abstract

With the increasing popularity of species identification smartphone apps, citizen scientists contribute to large and rapidly growing vegetation data collections. The question emerges whether such data can be utilized to monitor essential biodiversity variables across the globe.

Here, we use the freely available field observations of vascular plants provided by iNaturalist, a citizen science project that has encouraged users across the globe to identify, share and jointly validate species they encounter via photo and geolocation. We test whether iNaturalist observations complemented with trait measurements from the TRY database (Kattge et al, 2020) are able to represent global trait patterns.

As a reference for evaluating the iNaturalist observations, we use trait community-weighted means from the database sPlotOpen (Bruelheide et al, 2019; Sabatini et al, 2021). sPlotOpen is a curated database of globally distributed plots with vegetation abundance measurements, balanced over global climate and soil conditions. It provides community-weighted means for each vegetation plot for 18 traits. These community-weighted means are also derived from TRY measurements. We thus compare spatially and taxonomically biased occurrence samples provided by iNaturalist citizen scientists to professionally sampled environmentally balanced plot-based abundance data.


## Outline
1. Preprocessing
    - Preprocessing iNaturalist observation data
    - Create TRY summary statistics per species
    - Linking iNaturalist and TRY via species name
    - Preprocessing vegetation plot data (sPlotOpen)
2. Make trait maps
3. Compare sPlotOpen and iNaturalist trait maps
4. Density of observations/plots in climate space
5. Spatial density vs. Difference
6. Differences among biomes
7. Life forms coverage
8. Compare sPlotOpen to published trait maps
9. Alternative approach: Aggregating observaions in buffers
    - Aggregate iNaturalist in buffer around sPlots
    - Correlation of buffer means



## Requirements

The following packages are needed for this workflow. Each subsection lists the the packages needed for that section only:

**Python packages**:

For handling data frames and (multidimensional) arrays:
  - ```pandas```
  - ```numpy```
  - ```xarray```

For handling geospatial data:
  - ```geopandas```
  - ```shapely```
  - ```rasterio```

For plotting:
  - ```matplotlib```
  - ```seaborn```
  - ```cartopy```
  - ```pyproj```

For fuzzy matching:
  - ```rapidfuzz```

For statistics:
  - ```statsmodels```
  - ```pylr2```

**R packages**:

For handling rasters:
  - ```raster```
  - ```rgdal```

For SMA regression:  
  - ```smatr```
