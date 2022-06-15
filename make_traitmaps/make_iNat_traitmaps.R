library(raster)
library(rgdal)

# Load iNat Data
print("Load iNaturalist-TRY data in R...")
iNat <- read.csv('iNat_TRY_log.csv')
xy <- cbind(iNat$decimalLongitude, iNat$decimalLatitude)

# raster for a 0.5 and 2 degree resolution map
r05 <- raster(ncols = 720, nrows = 360)
r2 <- raster(ncols = 180, nrows = 90)

loop.vector <- 7:24 # loop over trait columns in sPlotOpen dataframe

# export exp(ln()) maps
print("Genereate 2 degree maps...")
for (i in loop.vector) { # Loop over loop.vector
  vals <- iNat[,i]
  name <- colnames(iNat[i])

  # 2 ln
  raster2 <- rasterize(xy, r2, vals, fun = mean)
  raster2[is.infinite(raster2)] <- NA
  crs(raster2) <- "+proj=longlat"
  filename = paste("iNat_", name, "_2deg_ln.tif", sep="")
  writeRaster(raster2,filename, overwrite=TRUE)

  # 2 exp
  raster2 <- exp(raster2)
  filename = paste("iNat_", name, "_2deg_expln.tif", sep="")
  writeRaster(raster2,filename, overwrite=TRUE)


}

print("Genereate 0.5 degree maps...")
for (i in loop.vector) { # Loop over loop.vector
  vals <- iNat[,i]
  name <- colnames(iNat[i])

  # 0.5 ln
  raster05 <- rasterize(xy, r05, vals, fun = mean)
  raster05[is.infinite(raster05)] <- NA
  crs(raster05) <- "+proj=longlat"
  filename = paste("iNat_", name, "_05deg_ln.tif", sep="")
  writeRaster(raster05,filename, overwrite=TRUE)

  # 0.5 exp
  raster05 <- exp(raster05)
  filename = paste("iNat_", name, "_05deg_expln.tif", sep="")
  writeRaster(raster05,filename, overwrite=TRUE)

}
