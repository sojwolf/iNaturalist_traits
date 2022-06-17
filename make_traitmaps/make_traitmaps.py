import pandas as pd # for handling dataframes in python
import numpy as np # array handling
import getopt, sys
import os


#18:56

iNat_filename = ''
TRY_filename = ''

try:
    opts, args = getopt.getopt(sys.argv[1:],"hi:t:",["iNat=","TRY="])
except getopt.GetoptError:
    print('Error: use ')
    sys.exit(2)
for opt, arg in opts:
    if opt == '-h':
        print('python make_traitmaps.py -n iNaturalist_filename -t TRY_filename')
        sys.exit()
    elif opt in ("-i", "--iNat"):
        iNat_filename = arg
    elif opt in ("-t", "--TRY"):
        TRY_filename = arg

print('iNat file: ', iNat_filename)
print( 'TRY file: ', TRY_filename)



#################################
######## Load iNat Data #########
#################################
print("Loading iNaturalist data...")
# read file name
iNat = pd.read_csv(iNat_filename, sep='\t')

iNat = iNat[["gbifID", "scientificName","decimalLatitude","decimalLongitude",
            "eventDate", "dateIdentified"]]

iNat['scientificName']  = iNat['scientificName'].apply(lambda x: ' '.join(x.split()[0:2]))

##########iNat.to_csv("Data/iNat/observations.csv", index=False)

#################################
###### TRY summary stats ########
#################################

print("Preprocessing TRY data...")
TRYdata = pd.read_csv(TRY_filename, sep = "\t", encoding="iso-8859-1",
                     usecols = ["AccSpeciesName", "SpeciesName", "TraitID", "TraitName", "StdValue"],
                     dtype={'TraitID': float})

# make all letters lower case
TRYdata['AccSpeciesName'] = TRYdata['AccSpeciesName'].str.lower()
# capitalize first letter in string
TRYdata['AccSpeciesName'] = TRYdata['AccSpeciesName'].str.capitalize()
# get only two first words (split at space)
TRYdata['AccSpeciesName']  = TRYdata['AccSpeciesName'].apply(lambda x: ' '.join(x.split()[0:2]))
# change type to string
TRYdata['AccSpeciesName'] = TRYdata['AccSpeciesName'].astype(str)

# same for species name
TRYdata['SpeciesName'] = TRYdata['SpeciesName'].str.lower()
TRYdata['SpeciesName'] = TRYdata['SpeciesName'].str.capitalize()
TRYdata['SpeciesName'] = TRYdata['SpeciesName'].astype(str)
TRYdata['SpeciesName']  = TRYdata['SpeciesName'].apply(lambda x: ' '.join(x.split()[0:2]))

# group data by species name and trait
grouped = TRYdata.groupby(['AccSpeciesName', 'TraitID', 'TraitName'])
TRYsummary = grouped['StdValue'].agg([np.mean]).reset_index()

def shorten_names(df):

    df.rename(columns = {'Stem specific density (SSD) or wood density (stem dry mass per stem fresh volume)':'SSD'}, inplace = True)
    df.rename(columns = {'Leaf carbon (C) content per leaf dry mass':'Leaf C'}, inplace = True)
    df.rename(columns = {'Leaf nitrogen (N) content per leaf dry mass':'Leaf N per mass'}, inplace = True)
    df.rename(columns = {'Leaf phosphorus (P) content per leaf dry mass':'Leaf P'}, inplace = True)
    df.rename(columns = {'Leaf dry mass per leaf fresh mass (leaf dry matter content, LDMC)':'LDMC'}, inplace = True)
    df.rename(columns = {'Seed dry mass':'Seed mass'}, inplace = True)
    df.rename(columns = {'Seed length':'Seed length'}, inplace = True)
    df.rename(columns = {'Leaf nitrogen (N) content per leaf area':'Leaf N per area'}, inplace = True)
    df.rename(columns = {'Leaf nitrogen/phosphorus (N/P) ratio':'Leaf N P ratio'}, inplace = True)
    df.rename(columns = {'Leaf nitrogen (N) isotope signature (delta 15N)':'Leaf delta15N'}, inplace = True)
    df.rename(columns = {'Leaf fresh mass':'Leaf fresh mass'}, inplace = True)
    df.rename(columns = {'Seed number per reproducton unit':'Seeds per rep. unit'}, inplace = True)
    df.rename(columns = {'Stem conduit density (vessels and tracheids)':'Stem conduit density'}, inplace = True)
    df.rename(columns = {'Dispersal unit length':'Dispersal unit length'}, inplace = True)
    df.rename(columns = {'Wood vessel element length; stem conduit (vessel and tracheids) element length':'Conduit element length'}, inplace = True)
    df.rename(columns = {'Plant height vegetative':'Plant Height'}, inplace = True)
    df.rename(columns = {'Leaf area (in case of compound leaves: leaflet, undefined if petiole is in- or excluded)':'Leaf Area'}, inplace = True)
    df.rename(columns = {'Leaf area per leaf dry mass (specific leaf area, SLA or 1/LMA): undefined if petiole is in- or excluded':'SLA'}, inplace = True)

TRY = TRYsummary.pivot(index=["AccSpeciesName"], columns="TraitName", values="mean")

# reset indeces (species name) as columns in data frame
TRY.reset_index(inplace=True)
shorten_names(TRY)

#########TRYsummary_t.to_csv("TRY/TRY_summary_stats.csv", index=False)

# same with original TRY name
# group data by species name and trait, same analysis as above
grouped_syn = TRYdata.groupby(['SpeciesName', 'TraitID', 'TraitName'])

TRYsummary_syn = grouped_syn['StdValue'].agg([np.mean]).reset_index()

# change df shape
TRY_syn = TRYsummary_syn.pivot(index=["SpeciesName"], columns="TraitName", values="mean")

# reset indeces (species name) as columns in data frame
TRY_syn.reset_index(inplace=True)

# shorten column names
shorten_names(TRY_syn)

############# TRYsummary_t_syn.to_csv("TRY/TRY_summary_stats_syn.csv", index=False)

############################################
######## Link iNaturalist and TRY ##########
############################################

print("Linking iNaturalist and TRY...")

iNat_TRY = pd.merge(iNat, TRY,
                    left_on= ['scientificName'],
                    right_on= ['AccSpeciesName'],
                    how='inner')

# filter for observations not in merged dataframe:
iNat_rest = iNat[~iNat.gbifID.isin(iNat_TRY['gbifID'])]

# non-fuzzy merge with TRY summary stats on original TRY species name:

iNat_TRY_syn = pd.merge(iNat_rest, TRY_syn,
                    left_on= ['scientificName'],
                    right_on= ['SpeciesName'],
                    how='inner')

subsets = [iNat_TRY, iNat_TRY_syn]

iNat_TRY_all = pd.concat(subsets)
iNat_TRY_all = iNat_TRY_all.drop(['AccSpeciesName', 'SpeciesName'], axis = 1)

trait = iNat_TRY_all.columns[6:24]

iNat_TRY_all = iNat_TRY_all.replace(-np.inf, np.nan)
iNat_TRY_all = iNat_TRY_all.replace(np.inf, np.nan)

iNat_TRY_all.loc[:, trait] = np.log(iNat_TRY_all[trait])

print("Exporting linked iNaturalist-TRY data frame to csv...")
iNat_TRY_all.to_csv("iNat_TRY_log.csv", index=False)

###########################
##### make GeoTiffs #######
###########################

os.system('Rscript make_iNat_traitmaps.R')
