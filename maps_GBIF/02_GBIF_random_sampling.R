library(tidyverse)
library(data.table)
library(vegan)
#library(igraph)
library(parallel)
library(dplyr)

#set up data.table and blas threads to not overload the cluster
dtn = 30
setDTthreads(threads = dtn)

#====01. Perform random sampling================================================
file <- "/net/scratch/swolf/GBIF/01_gbif_species_2023-06-07.csv"

#----GBIF-----------------------------------------------------------------------
full_dt <- fread(file)

#filter for species in TRY gap-filled data
TRY <- fread("/net/scratch/swolf/TRY_5_gapfilled/TRY_50_2020_01/gapfilled_data/species_means.csv")
length(unique(TRY[["Species"]]))

head(full_dt)

# test extracting first two words
file_test <- "/net/scratch/swolf/GBIF/test_file.csv"
test <- fread(file_test)
str_extract(test$scientificName, '[A-Za-z]+ [A-Za-z]+')

#extract only first two words of species name
full_dt$speciesName <- str_extract(full_dt$scientificName, '[A-Za-z]+ [A-Za-z]+')
TRY$speciesName <- str_extract(TRY$Species, '[A-Za-z]+ [A-Za-z]+')
length(unique(TRY[["Species"]]))
length(unique(TRY[["speciesName"]]))
TRY <- TRY %>% drop_na(speciesName)
head(TRY)

#get accepted species names from species list
species_list <- fread("/net/data/GBIF/2023_05_30/0005633-230530130749713.csv")
head(species_list)
#merged_species <- merge(full_dt, species_list, by.x = "Species", by.y = "species")
dim(species_list)
length(unique(species_list[["acceptedScientificName"]]))
length(unique(species_list[["scientificName"]]))
length(unique(species_list[["speciesKey"]]))

# Join on different column names
merged_df <- merge(full_dt, TRY, by = "speciesName")
head(merged_df)
dim(merged_df)
dim(merged_df)/dim(full_dt) # 92% of observations matched to trait information
length(unique(merged_df[["speciesName"]])) # 36520 species matched
length(unique(merged_df[["speciesName"]]))/length(unique(full_dt[["speciesName"]])) # 24% of GBIF species matched
length(unique(merged_df[["speciesName"]]))/length(unique(TRY[["speciesName"]])) # 70% of species from TRY matched

#number of unique hexagons in entire dataframe
for(col_name in c("hex6","hex7","hex8","hex9","hex10")){
  print(length(unique(merged_df[[col_name]])))
}

#create downsample dataset for traitmaps
set.seed(1)
sample_all_cells_7 <- merged_df[, if(.N > 10000) .SD[sample(x = .N, size = 10000, replace=F)] else .SD[sample(x = .N, size = .N)], by = "hex7"]
length(unique(sample_all_cells_7[["hex7"]]))
fwrite(sample_all_cells_7, paste0("/net/scratch/swolf/GBIF/gbif_sample_all_cells_hex7.csv"), row.names = F)
dim(sample_all_cells_7)

set.seed(2)
sample_all_cells_9 <- merged_df[, if(.N > 10000) .SD[sample(x = .N, size = 10000, replace=F)] else .SD[sample(x = .N, size = .N)], by = "hex9"]
length(unique(sample_all_cells_9[["hex9"]]))
fwrite(sample_all_cells_9, paste0("/net/scratch/swolf/GBIF/gbif_sample_all_cells_hex9.csv"), row.names = F)
dim(sample_all_cells_9)

#downsample only from well-sampled hexagons
for(col_name in c("hex6","hex7","hex8","hex9","hex10")){
  for(i in c(78, 156, 312, 625, 1250, 2500, 5000)){
    for(j in 1:5){
      set.seed(j)
      sample <- merged_df[, if(.N > 5000) .SD[sample(x = .N, size = i, replace=F)], by = col_name]
      print(length(unique(sample[[col_name]])))
      fwrite(sample, paste0("/net/scratch/swolf/GBIF/gbif_samples_5000/gbif_sample_", col_name, "_n", i, "_s", j, ".csv"), row.names = F)
    }
  }
}

#downsample min. of sample size
for(col_name in c("hex6","hex7","hex8","hex9","hex10")){
  for(i in c(78, 156, 312, 625, 1250, 2500, 5000)){
    for(j in 1:5){
      set.seed(j)
      sample <- merged_df[, if(.N > i) .SD[sample(x = .N, size = i, replace=F)], by = col_name]
      print(length(unique(sample[[col_name]])))
      fwrite(sample, paste0("/net/scratch/swolf/GBIF/gbif_samples_flex/gbif_sample_", col_name, "_n", i, "_s", j, ".csv"), row.names = F)
    }
  }
}

