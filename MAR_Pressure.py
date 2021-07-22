#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 28 13:02:04 2021

@author: gomez1mi
"""

import numpy as np
import xarray as xr
import pickle
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.animation import FuncAnimation

#flag to export npy files
export = False 

#flag to filter out years with start/end years. Set False when comparing ERA5 and MPI
full = True 

#flag to calculate SLP using alternate methods (i.e., SLP2 and SLP3)
alter = False 

#flag to produce the animation
animate = False 

varname = 'SP'
ERA_path = '/.fsnet/project/meige/2021/21DATMAROBSG/MAR/MAR-ERA5/'
MPI_path = '/.fsnet/project/meige/2021/21DATMAROBSG/MAR/MAR-MPI-SSP5/'
ERA_filepath = f'{ERA_path}*.{varname}.*.nc'
MPI_filepath = f'{MPI_path}*.{varname}.*.nc'

ERA_ST_fp = f'{ERA_path}*.ST.*.nc'
ERA_Q_fp = f'{ERA_path}*.QQz.*.nc'
ERA_TT_fp = f'{ERA_path}*.TTz.*.nc'


MPI_ST_fp = f'{MPI_path}*.ST.*.nc'
MPI_Q_fp = f'{MPI_path}*.QQz.*.nc'

#Load SP and other variables
ERA_ds = xr.open_mfdataset(ERA_filepath)
MPI_ds = xr.open_mfdataset(MPI_filepath)

if alter:
    ERA_ST = xr.open_mfdataset(ERA_ST_fp)['ST']
    MPI_ST = xr.open_mfdataset(MPI_ST_fp)['ST']
    ERA_Q = xr.open_mfdataset(ERA_Q_fp).sel(ztqlev=2)['QQz']
    MPI_Q = xr.open_mfdataset(MPI_Q_fp).sel(ztqlev=2)['QQz']
    ERA_T2 = xr.open_mfdataset(ERA_TT_fp).sel(ztqlev=2)['TTz']


#loading surface heights
ERA_SH = xr.open_mfdataset(r'/.fsnet/project/meige/2021/21DATMAROBSG/MAR/*ERA5*.nc').SH
MPI_SH = xr.open_mfdataset(r'/.fsnet/project/meige/2021/21DATMAROBSG/MAR/*MPI*.nc').SH



#ERA5 Month Filter. Note all ERA5 variables have same time series
ERA_m = ERA_ds['time.month']
ERA_ds = ERA_ds.where(np.logical_or(ERA_m<=4, ERA_m>=10), drop = True)
if alter:
    ERA_ST = ERA_ST.where(np.logical_or(ERA_m<=4, ERA_m>=10), drop = True)
    ERA_Q = ERA_Q.where(np.logical_or(ERA_m<=4, ERA_m>=10), drop = True)

#MPI month Filter
MPI_m = MPI_ds['time.month']
MPI_ds = MPI_ds.where(np.logical_or(MPI_m<=4, MPI_m>=10), drop = True)
if alter:
    MPI_ST = MPI_ST.where(np.logical_or(MPI_m<=4, MPI_m>=10), drop = True)
    MPI_Q = MPI_Q.where(np.logical_or(MPI_m<=4, MPI_m>=10), drop = True)


if not full:
    #Period for which to check data
    syear = 1981 #start year
    eyear = 2018 #end year
    
    ERA_y = ERA_ds['time.year']
    ERA_ds = ERA_ds.where(np.logical_and(ERA_y<=eyear, ERA_y>=syear), drop = True)
    if alter:
        ERA_ST = ERA_ST.where(np.logical_and(ERA_y<=eyear, ERA_y>=syear), drop = True)
        ERA_Q = ERA_Q.where(np.logical_and(ERA_y<=eyear, ERA_y>=syear), drop = True)
        
    MPI_y = MPI_ds['time.year']
    MPI_ds = MPI_ds.where(np.logical_and(MPI_y<=eyear, MPI_y>=syear), drop = True)
    if alter:
        MPI_ST = MPI_ST.where(np.logical_and(MPI_y<=eyear, MPI_y>=syear), drop = True)
        MPI_Q = MPI_Q.where(np.logical_and(MPI_y<=eyear, MPI_y>=syear), drop = True)


#scale height given by R * T / g, ~8000 with R = 287 J/(kg K), 
#g = 9.81 m / s**2, T = 273 K
scale_height = 7987
ERA_SLP = ERA_ds['SP'] * np.exp(ERA_SH/scale_height)
MPI_SLP = MPI_ds['SP'] * np.exp(MPI_SH/scale_height)


#%%SLP v2, v3 Calc
if alter:
    #calculate water mixing ratio from Q
    ERA_w = ERA_Q * 1e-3 / (1 - ERA_Q * 1e-3)
    MPI_w = MPI_Q * 1e-3 / (1 - MPI_Q * 1e-3)
    
    #calculate the partial vapor pressure Using US-ARMY HANDBOOK from 1975
    ERA_e = (ERA_w * ERA_ds['SP']) / (0.622 + ERA_w)
    MPI_e = (MPI_w * MPI_ds['SP']) / (0.622 + MPI_w)
    
    #calculate the density of dry air, R_dry = 287.058
    R_dry = 287.058
    
    #Julien Constants
    g = 9.81 #m/s**2
    a = 6.5e-3 #adiabatic lapse rate free atmosphere, K/m
    Ch = 0.15 #factor from Julien. ???
    
    ERA_SLP2 = ERA_ds['SP'] * np.exp( (g / (R_dry * ( (ERA_ST + 273.15) + Ch * ERA_e + 0.5 * a * ERA_SH)) * ERA_SH) )
    MPI_SLP2 = MPI_ds['SP'] * np.exp( (g / (R_dry * ( (MPI_ST + 273.15) + Ch * MPI_e + 0.5 * a * MPI_SH)) * MPI_SH) )
    
    ERA_e2 = ERA_Q * 1e-3 * ERA_ds['SP'] / 0.622 #Gialotti
    MPI_e2 = MPI_Q * 1e-3 * MPI_ds['SP'] / 0.622 #Gialotti
    
    ERA_SLP3 = ERA_ds['SP'] * np.exp( (g / (R_dry * ( (ERA_ST + 273.15) + Ch * ERA_e2 + 0.5 * a * ERA_SH)) * ERA_SH) )
    MPI_SLP3 = MPI_ds['SP'] * np.exp( (g / (R_dry * ( (MPI_ST + 273.15) + Ch * MPI_e2 + 0.5 * a * MPI_SH)) * MPI_SH) )


#%%Time Average Calc
ERA_SLP_mean = ERA_SLP.mean(dim='time').values
MPI_SLP_mean = MPI_SLP.mean(dim='time').values
ERA_SP_mean = ERA_ds['SP'].mean(dim='time').values
MPI_SP_mean = MPI_ds['SP'].mean(dim='time').values

min_SP_val = np.min((ERA_SP_mean,MPI_SP_mean))
max_SP_val = np.max((ERA_SP_mean,MPI_SP_mean))


#%%Time average comparison plots
fig, axs = plt.subplots(1,2)
fig.suptitle('Time averaged Surface Pressure')
im1 = axs[0].imshow(ERA_SP_mean, vmin=min_SP_val, vmax=max_SP_val, cmap='jet', origin='lower')
axs[0].title.set_text('ERA5')
im2 = axs[1].imshow(MPI_SP_mean, vmin=min_SP_val, vmax=max_SP_val, cmap='jet', origin='lower')
axs[1].title.set_text('MPI SSP5')

# add space for colour bar
fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
fig.colorbar(im2, cax=cbar_ax)


min_SLP_val = np.min((ERA_SLP_mean,MPI_SLP_mean))
max_SLP_val = np.max((ERA_SLP_mean,MPI_SLP_mean))

fig, axs = plt.subplots(1,2)
fig.suptitle('Time averaged Sea Level Pressure')
im1 = axs[0].imshow(ERA_SLP_mean, vmin=min_SLP_val, vmax=max_SLP_val, cmap='jet', origin='lower')
axs[0].title.set_text('ERA5')
im2 = axs[1].imshow(MPI_SLP_mean, vmin=min_SLP_val, vmax=max_SLP_val, cmap='jet', origin='lower')
axs[1].title.set_text('MPI SSP5')

# add space for colour bar
fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
fig.colorbar(im2, cax=cbar_ax)

mean_diff = (MPI_SLP_mean - ERA_SLP_mean).mean()
#%%nice plot

fig, ax = plt.subplots(1,1,figsize=(5,2.5), dpi=200)
#fig.suptitle('Time averaged Surface Pressure')
im = ax.imshow(ERA_SP_mean, 
               vmin=ERA_SP_mean.min(), 
               vmax=ERA_SP_mean.max(), 
               cmap='jet', 
               origin='lower')
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
fig.colorbar(im, ax=ax)
fig.tight_layout()

#%% Export SLP
"""──────────────────────────────────────────────────────────────────────────┐
│In this section of the code, the SLP for MAR forced by MPI and MAR forced by│
│ERA5 are exported. Two normalization schemes are provided for MPI->MAR, via │
│ERA5 min and max values, and MPI min and max values up to a predefined      │
│limit, MPI_norm_year_limit.                                                 │
│                                                                            │
│Note that varname for saving ERA5 is different from that used in NN_data,   │
│can be switched back to SLPN if needed.                                     │
└──────────────────────────────────────────────────────────────────────────"""

if export:
    MPI_norm_year_limit = 2017
    save_path =  '/.fsnet/backup/save/gomez1mi/Python Scripts/NN_data/'
    
    era_min = ERA_SLP.min().values
    era_max = ERA_SLP.max().values
    era_recon = {
        'min':era_min,
        'range':era_max - era_min
        }
    
    era_rcpath = f'{save_path}SLPE_reconstructor.pkl'
    
    with open(era_rcpath,'wb') as handle:
        pickle.dump(era_recon,handle)
    
    mpi_temp = MPI_SLP2.where(MPI_SLP['time.year'] <= MPI_norm_year_limit, drop=True)
    mpi_min = mpi_temp.min().values
    mpi_max = mpi_temp.max().values
    mpi_recon = {
        'min':mpi_min,
        'range':mpi_max - era_min
        }
    
    mpi_rcpath = f'{save_path}SLPM_reconstructor.pkl'
    
    with open(mpi_rcpath,'wb') as handle:
        pickle.dump(mpi_recon,handle)
    
    
    for year in np.unique(ERA_SLP['time.year']):
        print(f'ERA {year}')
        
        t1 = np.logical_and(ERA_SLP['time.year'] == year, ERA_SLP['time.month'] >= 10)
        t2 = np.logical_and(ERA_SLP['time.year'] == year+1, ERA_SLP['time.month'] <= 4)
        t = np.logical_or(t1,t2)
        
        data = (ERA_SLP.where(t,drop=True) - era_recon['min']) / era_recon['range']
        
        filename = f'{save_path}SLPE_{year}.npy'
        
        np.save(filename, data.values)
    
    for year in np.unique(MPI_SLP['time.year']):
        print(f'MPI {year}')
        
        t1 = np.logical_and(MPI_SLP['time.year'] == year, MPI_SLP['time.month'] >= 10)
        t2 = np.logical_and(MPI_SLP['time.year'] == year+1, MPI_SLP['time.month'] <= 4)
        t = np.logical_or(t1,t2)
        
        data_e = (MPI_SLP.where(t,drop=True) - era_recon['min']) / era_recon['range']
        data_m = (MPI_SLP.where(t,drop=True) - mpi_recon['min']) / mpi_recon['range']
        
        filename_e = f'{save_path}SSEN_{year}.npy'
        filename_m = f'{save_path}SLPM_{year}.npy'
        
        np.save(filename_e, data_e.values)
        np.save(filename_m, data_m.values)

#%% Animation
"""──────────────────────────────────────────────────────────────────────────┐
│The animation code is used to provide a visual comparison between two field │
│variables. Note that the input is a set of data_arrays and not a set of     │
│datasets. The colors are scaled to cover the range of values, and the title │
│is currently set manually. This could be improved by reading the varname    │
│from the array.                                                              │
│                                                                            │
└──────────────────────────────────────────────────────────────────────────"""

def animate(xarrays, filename):

    cmap = "jet"
    
    norm = []
    norm.append(plt.Normalize(xarrays[0].min().values,xarrays[0].max().values))
    norm.append(plt.Normalize(xarrays[1].min().values,xarrays[1].max().values))
    
    cbar = []
    cbar.append(cm.ScalarMappable(norm=norm[0],cmap=cmap))
    cbar.append(cm.ScalarMappable(norm=norm[1],cmap=cmap))
    
    
    fig,axs = plt.subplots(1,2, figsize=(12,4))

    fig.colorbar(cbar[0], ax=axs[0])
    fig.colorbar(cbar[1], ax=axs[1])
    
    anim = FuncAnimation(fig, animation, interval=200, frames=xarrays[0].shape[0]-1, fargs=(xarrays,axs,norm,fig))
    anim.save(filename+'.mp4')
    plt.close(fig)

def animation(i, xarrays, axs, norm, fig):
    date = xarrays[0].time.isel(time=i).values.astype(str)[:10]
    fig.suptitle(f'{date}')
    for j in range(len(xarrays)):
        cmap='jet'
        axs[j].clear()
        if j == 0:
            axs[j].title.set_text('Sea Level Pressure')
        else:
            axs[j].title.set_text('Surface Pressure')
        axs[j].imshow(xarrays[j].isel(time=i).values, cmap=cmap, norm=norm[j], origin='lower')
        fig.tight_layout()
    if i%30 == 0: print(date)

if animate:
    d1 = ERA_SLP.where(ERA_SLP['time.year']==2006, drop=True)
    d2 = ERA_ds['SP'].where(ERA_ds['SP']['time.year']==2006, drop=True)
                                 
    xarrays = [d1, d2]
    filename = 'SLP_SP_compare'
    animate(xarrays, filename)

#%%Comparing SP in past and future - Signal
"""──────────────────────────────────────────────────────────────────────────┐
│This snippet of code produces the following plot:                           │
│A plot of the difference between 30-year averages between the past and      │
│future ( [2071-2100] minus [1981-2010] )                                    │
└──────────────────────────────────────────────────────────────────────────"""


#finding the indices corresponding to first 30 years in time series and
#last 30 years in time series
past_time = MPI_ds['time.year']<= np.unique(MPI_ds['time.year'])[30]
future_time = MPI_ds['time.year']>=np.unique(MPI_ds['time.year'])[-30]

#fitlering data
past_30 = MPI_ds.isel(time = past_time)
future_30 = MPI_ds.isel(time = future_time)

SP_past_mean = past_30['SP'].mean(dim=('time'))
SP_future_mean = future_30['SP'].mean(dim=('time'))

fig1, ax1 = plt.subplots()

(SP_future_mean - SP_past_mean).plot()
desc = '30 Year Average Surface Pressure Difference \n between (2071-2100) and (1981-2010) '
ax1.title.set_text(desc)


#%%Spatial Yearly mean calc
"""──────────────────────────────────────────────────────────────────────────┐
│This snippet of code produces the following plot:                           │
│Scatterplot of the yearly spatially-averaged surface pressure in the MAR    │
│region, as well as the line of best fit.                                    │
└──────────────────────────────────────────────────────────────────────────"""


#Surface Pressure
spatial_mean = MPI_ds.mean(dim=('x','y'))

yearly = spatial_mean.resample(time='1Y')

coeff = yearly.mean().polyfit('time',deg=1)
fit = xr.polyval(yearly.mean()['time'],coeff)

fig2, ax2 = plt.subplots()

xr.plot.scatter(yearly.mean(),'time','SP')
fit['SP_polyfit_coefficients'].plot(color='black', label = 'Fit')

plt.ylabel('Surface Pressure (Yearly Spatial Mean, hPa)')
plt.legend()
plt.tight_layout()

"""──────────────────────────────────────────────────────────────────────────┐
│This snippet of code produces the following plot:                           │
│Scatterplot of the yearly spatially-averaged sea level pressure in the MAR  │
│region, as well as the line of best fit.                                    │
└──────────────────────────────────────────────────────────────────────────"""
#Sea Level Pressure
spatial_mean_SLP = MPI_SLP.mean(dim=('x','y'))

yearly_SLP = spatial_mean_SLP.resample(time='1Y')
data = yearly_SLP.mean().to_dataset(name='SLP')

coeff_SLP = data['SLP'].polyfit('time',deg=1)
fit_SLP = xr.polyval(yearly_SLP.mean()['time'],coeff_SLP)

fig_SLP, ax_SLP = plt.subplots()

xr.plot.scatter(data,'time','SLP')
fit_SLP['polyfit_coefficients'].plot(color='black', label = 'Fit')

plt.ylabel('Sea Level Pressure (Yearly Spatial Mean, hPa)')
plt.legend()
plt.tight_layout()

#calculate linreg to get pval
lin_reg = stats.linregress(data['time.year'].values,data['SLP'].values)

ax_SLP.title.set_text(f"p value:{lin_reg.pvalue}")
plt.tight_layout()


#%% Grenoble analysis 
"""──────────────────────────────────────────────────────────────────────────┐
│This snippet of code produces the following plot:                           │
│Scatterplot of the yearly-averaged sea level pressure at grenoble in the MAR│
│simulation, as well as the line of best fit.                                │
└──────────────────────────────────────────────────────────────────────────"""
#Surface Pressure
grenoble = MPI_ds.isel(x=53,y=54)

yearly_g = grenoble.resample(time='1Y')
data = yearly_g.mean()

coeff_g = data.polyfit('time',deg=1, full=True)
fit2 = xr.polyval(data['time'],coeff_g)


fig2, ax2 = plt.subplots()

xr.plot.scatter(data,'time','SP')
fit2['SP_polyfit_coefficients'].plot(color='black', label = 'Fit')

plt.ylabel('Surface Pressure (Yearly Mean @ Grenoble: x=-112, y=70), hPa)')
plt.legend()


#calculate linreg to get pval
lin_reg = stats.linregress(data['time.year'].values,data['SP'].values)

ax2.title.set_text(f"p value:{lin_reg.pvalue}")
plt.tight_layout()

"""──────────────────────────────────────────────────────────────────────────┐
│This snippet of code produces the following plot:                           │
│Scatterplot of the yearly-averaged sea level pressure at grenoble in the MAR│
│simulation, as well as the line of best fit.                                │
└──────────────────────────────────────────────────────────────────────────"""
#Sea Level Pressure
grenoble_SLP = MPI_SLP.isel(x=53,y=54)

yearly_gSLP = grenoble_SLP.resample(time='1Y')
data_gSLP = yearly_gSLP.mean().to_dataset(name='SLP')

coeff_gSLP = data_gSLP.polyfit('time',deg=1, full=True)
fit_gSLP = xr.polyval(data_gSLP['time'],coeff_gSLP)


fig_gSLP, ax_gSLP = plt.subplots()

xr.plot.scatter(data_gSLP,'time','SLP')
fit_gSLP['SLP_polyfit_coefficients'].plot(color='black', label = 'Fit')

plt.ylabel('SLP (Yearly Mean @ Grenoble: x=-112, y=70), hPa)')
plt.legend()


#calculate linreg to get pval
lin_reg = stats.linregress(data_gSLP['time.year'].values,data_gSLP['SLP'].values)

ax_gSLP.title.set_text(f"p value:{lin_reg.pvalue}")
plt.tight_layout()


