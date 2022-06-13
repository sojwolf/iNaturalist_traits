# Citizen science plant observations encode global trait patterns

This repository hosts additional material accompanying the manuscript "Citizen science plant observations encode global trait patterns":

  1. The entire workflow documentaion: https://sojwolf.github.io/iNaturalist_traits/
  2. GeoTiff trait maps based on iNaturalist observations and sPlotOpen observaitons, respectively

## Trait maps

### Versions

1. iNaturalist

        1. 0.5 degrees

              1. ln
              2. exp(ln)

        2. 2 degrees

              1. ln
              2. exp(ln)

2. sPlotOpen

        1. 0.5 degrees

              1. ln
              2. exp(ln)

        2. 2 degrees

              1. ln
              2. exp(ln) 

### Load in Python

Get file names:

```python
from os import listdir
from os.path import isfile, join

path = "iNaturalist_traits-main/iNat_maps/2_deg/ln/"
files = [f for f in listdir(path) if isfile(join(path, f))]
files.sort()
```
Load all trait maps as xarray:
```python
from import xarray as xr

def cubeFile(file):
    name = file.replace(".tif","")
    sr = xr.open_dataset(path + file,engine = "rasterio",chunks = 1024).sel(band = 1)
    sr = sr.assign_coords({"variable":name})
    return sr

da = xr.concat([cubeFile(x) for x in files],dim = "variable")
```
Select a specific band:
```
da.band_data.sel(variable = "iNat_Leaf.Area_2_ln")
```
Convert to data frame

```python
df = da_2.band_data.to_dataset().to_dataframe().reset_index()
df_spread = df.pivot(index= ['x','y'],columns='variable',values='band_data').reset_index()
```


### Load in R
