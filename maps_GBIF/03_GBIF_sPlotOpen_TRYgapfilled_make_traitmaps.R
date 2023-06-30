library(raster)
library(data.table)
library(terra)
#Load Data

dtn = 30
setDTthreads(threads = dtn)

file <- "/net/scratch/swolf/GBIF/gbif_sample_all_cells_hex9.csv" #random sample max. 10000 obs, hex9, GBIF, TRY-gapfilled
 
data <- fread(file)
#data <- read.csv(file)

head(data)
xy <- cbind(data$decimalLongitude, data$decimalLatitude)

# raster for a 2 degree resolution map
r02 <- raster(ncols = 1800, nrows = 900)
r05 <- raster(ncols = 720, nrows = 360)
r2 <- raster(ncols = 180, nrows = 90)

rasters <- c(r02, r05, r2)
#rasters <- c(r05)
loop.vector <- 15:47 # loop over these trait columns in dataframe
#loop.vector <- 15:15 # loop over these trait columns in dataframe
folder_name <- c("02deg","05deg","2deg")
#folder_name <- c("05deg")
#folder_name <- c("05deg")

index <- 1
for (j in rasters) {
  
  for (i in loop.vector) { # Loop over loop.vector
    vals <- data[,i, with=FALSE]
    name1 <- colnames(data[,i, with=FALSE])
    #r1 <- rasterize(xy, j, vals, fun = mean)
    r1 <- rasterize(xy, j, vals, fun=function(x,...)c(length(x),mean(x),median(x),sd(x)))
    r1[is.infinite(r1)] <- NA
    crs(r1) <- "+proj=longlat"
    
    names(r1)<-c("count", "mean", "median", "sd")
    #export as GeoTiff -->  separate file for each trait
    
    filename1 = paste("iNaturalist/FuncDiv/traitmaps/TRY_gap_filled/more_layers/", folder_name[index],  "/GBIF_TRYgapfilled_", name1, "_", folder_name[index], ".tif", sep="")
    print(filename1)
    writeRaster(r1, filename1, overwrite=TRUE, format="raster")
  
  }
  
  index <- index +1
}  

#test
test <- brick("iNaturalist/FuncDiv/traitmaps/TRY_gap_filled/more_layers/2deg/GBIF_TRYgapfilled_X1080_2deg.gri")
test
plot(test)

##################

library(raster)
library(data.table)
#Load Data

dtn = 30
setDTthreads(threads = dtn)

file <- "/net/home/swolf/iNaturalist/Data/sPlotOpen/sPlotOpen_TRYgapfilled_cwm.csv" #sPlotOpen gap-filled trait maps

data <- fread(file)
#data <- read.csv(file)

head(data)
xy <- cbind(data$Longitude, data$Latitude)

# raster for a 2 degree resolution map
r02 <- raster(ncols = 1800, nrows = 900)
r05 <- raster(ncols = 720, nrows = 360)
r2 <- raster(ncols = 180, nrows = 90)

rasters <- c(r02, r05, r2)
loop.vector <- 2:34 # loop over these trait columns in dataframe
folder_name <- c("02deg","05deg","2deg")
#folder_name <- c("05deg")

index <- 1
for (j in rasters) {
  
  for (i in loop.vector) { # Loop over loop.vector
    vals <- data[,i, with=FALSE]
    name1 <- colnames(data[,i, with=FALSE])
    r1 <- rasterize(xy, j, vals, fun = mean)
    r1[is.infinite(r1)] <- NA
    crs(r1) <- "+proj=longlat"
    
    #export as GeoTiff -->  separate file for each trait
    
    filename1 = paste("iNaturalist/FuncDiv/traitmaps/TRY_gap_filled/", folder_name[index],  "/sPlotOpen_TRYgapfilled_", name1, "_", folder_name[index], ".tif", sep="")
    print(filename1)
    writeRaster(r1, filename1, overwrite=TRUE)
  }
  
  index <- index +1
}  
