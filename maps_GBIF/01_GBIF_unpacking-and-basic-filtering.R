# Copied from Daria ~/iNaturalist/FuncDiv/01_prep-gbif_unpacking-and-basic-filtering.R

library(data.table)
library(tidyverse)
library(dggridR)
library(sf)
library(giscoR)
library(CoordinateCleaner)
#library(rgbif) #we only need this for data request via R

dtn = 30
setDTthreads(threads = dtn) #defines how many threads data.table is allowed to use
#getDTthreads(verbose=TRUE)

#----1. GBIF download-----------------------------------------------------------
# We can omit this step since we have prepared the data download using GBIF
# visual interface because it gives better control over the parameters


# fill in your gbif.org credentials 
# user  <- ""                 # your gbif.org username
# pwd   <- ""    # your gbif.org password
# email <- ""  # your email

# GBIF API parameters https://www.gbif.org/developer/summary
# spin up a download request for GBIF occurrence data

#occ_download() approach does not allow to use IUCN categories
#for this reason download is prepared via visual editor on the website
# gbif_vasc <- occ_download(
#   pred_and(
#     pred("occurrenceStatus", "PRESENT"), pred("taxonKey", 7707728),
#     pred("hasCoordinate", TRUE), pred("hasGeospatialIssue", FALSE),
#     pred_in("basisOfRecord", c("OBSERVATION", "MACHINE_OBSERVATION", "HUMAN_OBSERVATION",
#                              "OCCURENCE", "MATERIAL_SAMPLE", "PRESERVED_SPECIMAN",
#                              "MATERIAL_CITATION"))
#        ),
#   format = "SIMPLE_CSV",
#   user=user,pwd=pwd,email=email
# )


######## Download #########
# GBIF.org (18 May 2023) GBIF Occurrence Download https://doi.org/10.15468/dl.5a373r
#url_gbif <- "https://api.gbif.org/v1/occurrence/download/request/0247586-230224095556074.zip"
# unpack in terminal because the file is large (over 4 GB)
# /net/data/GBIF/2023_05_18/0247586-230224095556074.csv

#----2. Exploring and basic filtering of the data-------------------------------

file_csv <- "/net/data/GBIF/2023_05_30/0005636-230530130749713.csv"
file_csv <- "/net/scratch/swolf/GBIF/test_file.csv"
# read first 100 records to look at the data
# test_gbif_dt <- fread(file_csv, nrow = 100)

# to distinguish between unique species we use speciesKey as they refer to the AcceptedUsageKey in case of the synonyms
# using GBIF backbone taxonomy
# ?name_backbone_checklist for more information on how it works
# define the list of fields that will be used for filtering and further work
columns <- c("gbifID", "taxonRank",  "speciesKey", "year",
             "decimalLatitude", "decimalLongitude",
             "coordinateUncertaintyInMeters",  "basisOfRecord", "scientificName", 
             "datasetKey", "eventDate", "coordinatePrecision", "countryCode")
# some additional columns  useful for filtering may include
# "datasetKey", "species", "individualCount", "coordinatePrecision",
# "locality", "stateProvince", "issue"

gbif_dt <- fread(file_csv, select = columns) |> #nrow = 10000
  #countrycode(gbif_dt$countryCode, origin =  'iso2c', destination = 'iso3c') |>
  filter(taxonRank == 'SPECIES') |> #to make sure we only include records identified to species level
  filter(!is.na(speciesKey)) |> #to filter out records without keys
  #filter(basisOfRecord !="MATERIAL_SAMPLE" | basisOfRecord !="MATERIAL_CITATION") |> #already filtered in api #to filter out (usually) metagenomic records which can be quite unprecise
  filter(!coordinateUncertaintyInMeters %in% c(301, 3036, 999, 9999)) |> #to exclude errors that produced by geocoding software and do not represent real uncertainty values
  filter(coordinateUncertaintyInMeters < 10000 | is.na(coordinateUncertaintyInMeters)) |> # we do not filter NA's out by coordinateUncertainty and coordinatePrecision since these parameters are often omitted in newer datasets which rely on GPS obsservations
  filter(coordinatePrecision < 0.01 | is.na(coordinatePrecision)) |>
  filter(!decimalLatitude == 0 | !decimalLongitude == 0) |> #to exclude records along equator or central meridian
  #filter(year > 1900 |  is.na(year)) |> #already filterd in api #to exclude historical records that may fall outside Linnean taxonomy or wrong political borders
  cc_cen(buffer = 1000, lon = "decimalLongitude", lat = "decimalLatitude") |> # remove country centroids within 1km 
  cc_cap(buffer = 1000, lon = "decimalLongitude", lat = "decimalLatitude") |> # remove capitals centroids within 1km
  cc_inst(buffer = 1000, lon = "decimalLongitude", lat = "decimalLatitude") |> # remove zoo and herbaria within 1km 
  cc_sea(lon = "decimalLongitude", lat = "decimalLatitude") |> # remove from ocean 
  select(gbifID, speciesKey, scientificName, decimalLatitude, decimalLongitude, datasetKey, eventDate)

#with filtering for sea and centroids
dim(gbif_dt)
#231,982,614 observations
length(unique(gbif_dt[["speciesKey"]]))
#139,000 species by keys

#----3. Adding hexagonal grid seqnum to the coordinates-------------------------
# creating a grid with cell size of 23,323 sq. km
# with a mean distance between cell centrids of 165 km
# this roughly represents cartesian coordinates with resolution of 1.5 degree on equator
#if (!file.exists("output/01_dgg_hex_res07.rds")){
#  grid <- dgconstruct(res = 7)
#  saveRDS(grid, "output/01_dgg_hex_res07.rds")
#} else {
#  grid <- readRDS("output/01_dgg_hex_res07.rds")
#}

hex_size = seq(6,10)
cell_num = c(7292, 21872, 65612, 196832, 590492)

for (i in 1:length(hex_size)){
  print(hex_size[i])
  grid <- dgconstruct(res = hex_size[i])
  saveRDS(grid, paste0("/net/scratch/swolf/GBIF/01_dgg_hex_res", hex_size[i], ".rds"))
  
  #this is the simplest and the fastest approach instead of using parallel
  #system.time(
  grid_id <- dgGEO_to_SEQNUM(grid, gbif_dt$decimalLongitude, gbif_dt$decimalLatitude)
  
  #this takes less than an hour ~55 minutes
  #)
  col_name <- paste0("hex", as.character(hex_size[i]))
  gbif_dt[, col_name] <- grid_id$seqnum
  
  #to save grid as sf for further analysis and visualization

  grid_sf <- dgcellstogrid(grid, cells = 1:cell_num[i], savegrid = NA)  
  saveRDS(grid_sf, paste0("/net/scratch/swolf/GBIF/01_dgg_hex_res", hex_size[i], "_sf.rds"))
  
}

# save output GBIF data with grid id's in different resolutions
fwrite(gbif_dt, paste0("/net/scratch/swolf/GBIF/01_gbif_species_", Sys.Date(), ".csv"), row.names = F)

