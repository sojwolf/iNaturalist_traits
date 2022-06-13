#!/usr/bin/env python
# coding: utf-8

# # Buffers around sPlots

# For the alternative approach of using a buffer around each sPlotOpen vegetation plot, we aggregate all iNaturalist observations around each plot individually within a certain radius, or buffer, and calculate the mean trait measurement inside the buffer. 

# In[ ]:


import geopandas as gpd
import pandas as pd
import os
from csv import writer 


# ## Load and project data

# In[ ]:


iNat = pd.read_csv("iNat_TRY_log.csv")


# Covert into geopandas dataframes

# In[ ]:


geo_iNat = gpd.GeoDataFrame( iNat.iloc[:,:], geometry=gpd.points_from_xy(iNat.decimalLongitude, iNat.decimalLatitude), 
                            crs='epsg:4326')


# To obtain more equal area buffers, the latitudinal/longitudinal data is projected into ‘world sinusoidal projection’ (ESRI:54008).
# 
# See: https://gis.stackexchange.com/questions/383014/how-to-use-geopandas-buffer-function-to-get-buffer-zones-in-kilometers
# 

# In[ ]:


geo_iNat = geo_iNat.to_crs("ESRI:54008")


# ## Calculate iNaturalist values within buffer for a range of buffer sizes

# The iNaturalist observations within each buffer are aggregated and the average trait values calculated. 
# 
# - create bounding box for each plot  
# - then use r-tree to aggregate iNat observations
# 
# 
# See: https://geoffboeing.com/2016/10/r-tree-spatial-index-python/

# In[ ]:


# index iNaturalist
spatial_index = geo_iNat.sindex


# In[ ]:


# buffer size in meters
# for large buffersizes the run-time is substantial (days)

buffer_sizes = [1000,2000,4000,8000,16000,32000,64000,128000,256000]

for i in buffer_sizes:
    # directly update the point geometry with a polygon geometry for each sPlot
    # unit is same as lat/long
    # buffer 1 corresponds to 1 meter
    
    # load sPlot data, convert to geopandas dataframe, and reproject
    sPlot = pd.read_csv("sPlotOpen/cwm_loc.csv")
    geo_sPlot = gpd.GeoDataFrame(sPlot, geometry=gpd.points_from_xy(sPlot.Longitude, sPlot.Latitude), crs='epsg:4326')
    geo_sPlot = geo_sPlot.to_crs("ESRI:54008")
    geo_sPlot['geometry'] = geo_sPlot.geometry.buffer(i)
    
    # sPlot id
    
    filename = "Buffer_Rerun/all_buffer_means_" + str(i) + ".csv"
    
    # define output csv headers:

    column_names_iNat = iNat.columns[6:24]
    ids = pd.Index(["PlotObservationID"])
    obs_num = pd.Index(["NumberiNatObservations"])
    column_names = ids.append(column_names_iNat)
    column_names = obs_num.append(column_names)
    sPlot_buffers_means = pd.DataFrame(columns=column_names)
    sPlot_buffers_means.to_csv(filename,index=False)


    # for each sPlot, find matches in buffer and calculate iNat mean:

    for index, row in geo_sPlot.iterrows():
    
        # get intersection    
        geometry = row["geometry"]
    
        possible_matches_index = list(spatial_index.intersection(geometry.bounds))
        possible_matches = geo_iNat.iloc[possible_matches_index]

        precise_matches = possible_matches[possible_matches.intersects(geometry)]

        # calculate average values for each trait of iNaturalist observations inside buffer zone 
        
        means = precise_matches.mean()
        means['PlotObservationID'] = row['PlotObservationID']
        
        sPlot_buffers_means = pd.DataFrame(columns=column_names)
        sPlot_buffers_means = sPlot_buffers_means.append(means, ignore_index=True)
        sPlot_buffers_means = sPlot_buffers_means.astype(str) 
        sPlot_buffers_means['NumberiNatObservations'] = str(len(precise_matches))
    
        # write to file line by line
        sPlot_buffers_means.to_csv(filename,mode='a',index=False,header=False)


    
    

