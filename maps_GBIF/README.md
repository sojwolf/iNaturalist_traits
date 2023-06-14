# Trait maps based on sample of GBIF database

## Trait maps

Here you will find trait maps based on GBIF data and traits found in the TRY gap-filled dataset in GeoTIFF format at a 0.2°, 0.5°, and 2° resolutions. Each folder also contains the sPlotOpen maps for all respective traits and resolutions.

## Data

Source of species observations are GBIF sampled as such:
  1. GBIF download: https://doi.org/10.15468/dl.fe2kv3
  2. The observations were then linked to the TRY gap-filled dataset, which resulted in a total of n= observations. 90% of the GBIF observations were matched, 70% of species in TRY, and 24% of species in GBIF
  3. Matched observations were then binned into equal area hexagons (using the package size hex9, which corresponds to about 0.5 degrees at equator)
  4. From each hexagon were then sampled 10,000 observations. If a hexagon contained less than 10,000 observations, all observations were kept.
  5. This GBIF subsample contained approx. 35,000,000 observations

![Density GBIF](obs_density_GBIF_sample.PNG)

Figure 1: Global density of GBIF subsample at 2° resolution.

## Traits in TRY gap-filled:

## Correlation of sPlotOpen and GBIF sample for all traits at different resolutions

![Corr Plot](corr_res.PNG)
