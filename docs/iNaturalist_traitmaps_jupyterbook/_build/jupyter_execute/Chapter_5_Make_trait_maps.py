#!/usr/bin/env python
# coding: utf-8

# # Make GeoTiff trait maps  

# Here we create the trait maps and export them as GeoTiffs.
# 
# Note: **In this section we will use R because its provides more convinient raster processing pipelines.**

# In[ ]:


library(raster)


# In[ ]:


#Load sPlot Data
sPlot <- read.csv('sPlotOpen/cwm_loc.csv')
#Load iNat Data
iNat <- read.csv('iNat_TRY_log.csv')


# In[ ]:


xy_1 <- cbind(sPlot$Longitude, sPlot$Latitude)
xy_2 <- cbind(iNat$decimalLongitude, iNat$decimalLatitude)


# In[ ]:


# raster for a 2 degree resolution map

r <- raster(ncols = 180, nrows = 90)


# In[ ]:


loop.vector <- 5:22 # loop over trait columns in sPlotOpen dataframe

for (i in loop.vector) { # Loop over loop.vector
  vals_1 <- exp(sPlot[,i])
  name1 <- colnames(sPlot[i])
  r1 <- rasterize(xy_1, r, vals_1, fun = mean)
  r1[is.infinite(r1)] <- NA
  crs(r1) <- "+proj=longlat"
    
  vals_2 <- exp(iNat[name1])
  r2 <- rasterize(xy_2, r, vals_2, fun = mean)
  r2[is.infinite(r2)] <- NA
  crs(r2) <- "+proj=longlat"

  #export as GeoTiff -->  separate file for each trait

  filename1 = paste("sPlot_", name1, "_", deg, "deg.tif", sep="")
  writeRaster(r1,filename1, overwrite=TRUE)
  filename2 = paste("iNat_", name1, "_", deg, "deg.tif", sep="")
  writeRaster(r2,filename2, overwrite=TRUE)
}

