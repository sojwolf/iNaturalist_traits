#!/usr/bin/env python
# coding: utf-8

# # sPlotOpen Preprocessing
# 

# sPlotOpen (Sabatini et al, 2021) is an open-access and environmentally and spatially balanced subset of the global sPlot vegetation plots data set v2.1 (Bruelheide et al, 2019).
# 
# This section covers:
# 
# - Link plot coordinates with community wighted means (cwm)
# - Visualize plot density

# ## Download

# sPlotOpen Data is available at the *iDiv Data Repository*. For this study we used version 52.
# 
# https://idata.idiv.de/ddm/Data/ShowData/3474
# 
# 

# ## Packages

# In[ ]:


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm, Normalize
import cartopy.crs as ccrs
import cartopy as cart
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable


# ## Link plot coordinates to cmw data

# The data is stored in various tab-separated files:
# 
# - **sPlotOpen_header(2).txt** : contains information on each plot, such as coordinates, date, biome, country, etc.
# 
# 
# - **sPlotOpen_DT(1).txt** : contains information per plot and species with abundance and relative cover
# 
# 
# - **sPlotOpen_CWM_CWV(1).txt** : contains information on trait community weighted means and variances for each plot and 18 traits (ln-transformed)
# 

# In[5]:


# load the community weighted means

cwm = pd.read_csv("./sPlotOpen/sPlotOpen_CWM_CWV(1).txt", sep= "\t")


# In[6]:


plots = pd.read_csv("./sPlotOpen/sPlotOpen_header(2).txt", sep= "\t")


# In[7]:


sPlot = pd.merge(cwm, plots, on='PlotObservationID', how='inner')


# In[8]:


sPlot.head()


# ### Change columns name

# In[9]:


sPlot.columns


# In[10]:


# only variables not of interest have mixed data types
types = sPlot.applymap(type).apply(set)
types[types.apply(len) > 1]


# Rename the trait columns to match the TRY summary stats:

# In[11]:


sPlot.rename(columns = {'StemDens_CWM':'SSD'}, inplace = True)
sPlot.rename(columns = {'LeafC_perdrymass_CWM':'Leaf C'}, inplace = True)
sPlot.rename(columns = {'LeafN_CWM':'Leaf N per mass'}, inplace = True)
sPlot.rename(columns = {'LeafP_CWM':'Leaf P'}, inplace = True)
sPlot.rename(columns = {'LDMC_CWM':'LDMC'}, inplace = True)
sPlot.rename(columns = {'SeedMass_CWM':'Seed mass'}, inplace = True)
sPlot.rename(columns = {'Seed_length_CWM':'Seed length'}, inplace = True)
sPlot.rename(columns = {'LeafNperArea_CWM':'Leaf N per area'}, inplace = True)
sPlot.rename(columns = {'LeafNPratio_CWM':'Leaf N P ratio'}, inplace = True)
sPlot.rename(columns = {'Leaf_delta_15N_CWM':'Leaf delta15N'}, inplace = True)
sPlot.rename(columns = {'Leaffreshmass_CWM':'Leaf fresh mass'}, inplace = True)
sPlot.rename(columns = {'Seed_num_rep_unit_CWM':'Seeds per rep. unit'}, inplace = True)
sPlot.rename(columns = {'Stem_cond_dens_CWM':'Stem conduit density'}, inplace = True)
sPlot.rename(columns = {'Disp_unit_leng_CWM':'Dispersal unit length'}, inplace = True)
sPlot.rename(columns = {'Wood_vessel_length_CWM':'Conduit element length'}, inplace = True)
sPlot.rename(columns = {'PlantHeight_CWM':'Plant Height'}, inplace = True)
sPlot.rename(columns = {'LeafArea_CWM':'Leaf Area'}, inplace = True)
sPlot.rename(columns = {'SLA_CWM':'SLA'}, inplace = True)


# Replace infinite values with ```NaN```

# In[12]:


sPlot = sPlot.replace(-np.inf, np.nan)
sPlot = sPlot.replace(np.inf, np.nan)


# In[13]:


sPlot.head()


# Save this dataframe as ```csv```.

# In[12]:


sPlot.to_csv("sPlotOpen/cwm_loc.csv", index=False)


# ## Plot density

# In[16]:


def gridmap(long, lat, label, projection, colorbar=True):
    
    plt.rcParams.update({'font.size': 15})

    Z, xedges, yedges = np.histogram2d(np.array(long,dtype=float),
                                   np.array(lat),bins = [181, 91])

    #https://stackoverflow.com/questions/67801227/color-a-2d-histogram-not-by-density-but-by-the-mean-of-a-third-column
    #https://medium.com/analytics-vidhya/custom-strava-heatmap-231267dcd084
    
    #let function know what projection provided data is in:
    data_crs = ccrs.PlateCarree()
    
    #for colorbar
    cmap = plt.get_cmap('cool')
    im_ratio = Z.shape[0]/Z.shape[1]

    #plot map
    #create base plot of a world map
    ax = fig.add_subplot(1, 1, 1, projection=projection) # I used the PlateCarree projection from cartopy
    
    # set figure to map global extent (-180,180,-90,90)
    ax.set_global()
    
    #add coastlines
    ax.coastlines(resolution='110m', color='orange', linewidth=1.3)
    
    #add grid with values
    im = ax.pcolormesh(xedges, yedges, Z.T, cmap="cool", norm=LogNorm(), transform=data_crs, vmax = 400000)
    
    #add color bar
    if colorbar==True:
        fig.colorbar(im,fraction=0.046*im_ratio, pad=0.04, shrink=0.3, location="left", label=label)


# Apply the ```gridmap``` function to plot a 2D histogramm of the sPlotOpen plots and save output as ```.pdf```. You can also experiment with other projections. See https://scitools.org.uk/cartopy/docs/v0.15/crs/projections.html for inspiration.

# In[17]:


fig = plt.figure(figsize=(12, 12))
gridmap(sPlot['Longitude'], sPlot['Latitude'], "Number of sPlotOpen Plots", projection = ccrs.Robinson())
plt.savefig('../Figures/sPlot_density_Robinson_all.pdf', bbox_inches='tight')

