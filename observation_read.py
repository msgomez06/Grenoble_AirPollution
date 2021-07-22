#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 16:52:58 2021

This script is used to generate the temperature gradient files from the
observations available from Col de Porte, Pont de Claix, and Peuil de Claix

This script was written early on in the project and has not been refactored as
of July-22nd 2021.

@author: Milton Gomez
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import pandas as pd
import xarray as xr
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error


#%% Col de Porte dataload

#loading Col de Porte dataset - high altitude station
cdp_filename = 'FORCING_1993080106_2020080106_insitu.nc'
path = f'./obs_raw/{cdp_filename}'
cdp = xr.open_dataset(path)

#Select temperature and flag variables. Flag 1 = insitu, Flag 0 = SAFRAN
cdp = cdp[['Tair','flag']]


"""──────────────────────────────────────────────────────────────────────────┐
│Since Col de Porte data includes data from SAFRAN and it was found that     │
│temperature gradients calculated from SAFRAN are poorly correlted to those  │
│calculated from observations, we want to set SAFRAN values to NaN.          │
│                                                                            │
│Filtering out Safran:                                                       │
│                                                                            │
│drop = False ensures nan values where SAFRAN values used to be.             │
│                                                                            │   
│Note that interpolation is not necessary because CDP insitu measurements are│
│missing only during the summer! Thus, once measurements for a given winter  │
│begin there are no missing values between measurements.                     │
└──────────────────────────────────────────────────────────────────────────"""
cdp = cdp.where(cdp.flag == 1, drop = False)

#find daily mean temperature. Any hourly NaN will result in NaN daily mean.
cdp_dailyT = cdp.Tair.resample(time='1D', skipna=False).mean()

#%% Claix data loading
#loading Peuil and Pont de Claix stations (mid and low altitude, respectively)
claix_path = f"./obs_raw/temp_insitu_PodC_PedC_1985-2020.pkl"

with open(claix_path,'rb') as handle:
    claix = pickle.load(handle)
    
#interpolate to fill first forward NaN value
claix.interpolate(limit=1, limit_area='inside', inplace=True)

test = claix.copy()

#%% Data Healing
"""
Data healing:
    loading 2011-2017 station data for data healing. In this section, the 
    program finds a line of best fit for the T-T plot comparing Pont de Claix 
    with ILL, and Peuil de Claix with Les Ègaux
"""
#path = r'.\Data\InSituTemperatures.pkl'
heal_path = r'./obs_raw/temp_insitu_global.pkl'

with open(heal_path, 'rb') as handle:
    heal_data = pd.read_pickle(handle)

"""
Making a linear regressor for the low altitude data using ILL and for mid
altitude using St. Pierre les Egaux
"""

pont = claix['Pont_de_Claix']
peuil = claix['Peuil_de_Claix']

#Finding the non-nan times for ILL and St. Pierre
low_time1 = heal_data['ILL_T2'].dropna().index.values
mid_time1 = heal_data['ST-PIERRE-LESEGAUX'].dropna().index.values

low_time2 = pont.dropna().index.values
mid_time2 = peuil.dropna().index.values

low_index = np.intersect1d(low_time1, low_time2)
mid_index = np.intersect1d(mid_time1, mid_time2)

low_regressor = LinearRegression()
mid_regressor = LinearRegression()

low_regressor.fit(heal_data['ILL_T2'].loc[low_index], pont.loc[low_index])
mid_regressor.fit(heal_data['ST-PIERRE-LESEGAUX'].loc[mid_index], peuil.loc[mid_index])


#the time indices for which we have heal_data and data to be healed
low_val_times = np.intersect1d(claix.index.values, heal_data['ILL_T2'].dropna().index.values)
mid_val_times = np.intersect1d(claix.index.values, heal_data['ST-PIERRE-LESEGAUX'].dropna().index.values)

#find the indices where pont/peuil de claix have nan values
pont_nanidxs = np.where(pont.loc[low_val_times].isna())[0]
peuil_nanidxs = np.where(peuil.loc[mid_val_times].isna())[0]

#NaN values space variables
low_Ts = pont.loc[low_val_times].iloc[pont_nanidxs]
mid_Ts = peuil.loc[mid_val_times].iloc[peuil_nanidxs]

#calculating data to use to replace
low_data = low_regressor.predict(heal_data['ILL_T2'].dropna().iloc[pont_nanidxs])
mid_data = mid_regressor.predict(heal_data['ST-PIERRE-LESEGAUX'].dropna().iloc[peuil_nanidxs])

#replacing nans in dataframe
pont.loc[low_Ts.index] = low_data
peuil.loc[mid_Ts.index] = mid_data

#making x_arrays for ease of calculation; mirror variable T for ease of writing
low_values = pont.resample('1D').mean().to_xarray()
low_values['T'] = low_values[list(low_values.data_vars.keys())[0]]
mid_values = peuil.resample('1D').mean().to_xarray()
mid_values['T'] = mid_values[list(mid_values.data_vars.keys())[0]]

#%% Filemaking
#getting the timestamps for which the low-high dzt, low-mid dzt, and mid-high
#dzt values exist.
low_high_idxs = np.intersect1d(cdp_dailyT.get_index('time'),low_values.get_index('DATE'))
low_mid_idxs = np.intersect1d(low_values.get_index('DATE'),mid_values.get_index('DATE'))
mid_high_idxs = np.intersect1d(cdp_dailyT.get_index('time'),mid_values.get_index('DATE'))

#three loops for making winter files
savepath = './NN_data/' 

#low_high dzt loop:
data_high = cdp_dailyT.sel(time = low_high_idxs) - 273.15
data_low = low_values.T.sel(DATE = low_high_idxs)

#z @ col de porte: 1325 m ; z @ pont de claix: 237 m ; dz in km
dz = (1325 - 237)/1000

for year in np.unique(pd.DatetimeIndex(low_high_idxs).year):
    
    
    #t1 = np.logical_and(chartreuse['time.year']==year, chartreuse['time.month']>=10)
    t1 = np.logical_and(data_high['time.year']==year, data_high['time.month']>=10)
    t2 = np.logical_and(data_high['time.year']==year+1, data_high['time.month']<=4)
    t = data_high.time[np.logical_or(t1,t2)]
    
    t_high = data_high.sel(time = t).values
    t_low = data_low.sel(DATE = t).values.reshape(t_high.shape)
    
    dzt = (t_high - t_low)/dz
    
    #InSitu gradient - Low High (ISLH)
    filename = f'ISLH_{year}.npy'
    filepath = os.path.join(savepath,filename)
    
    np.save(filepath, dzt)


#low_mid dzt loop:
data_high = mid_values.T.sel(DATE = low_mid_idxs)
data_low = low_values.T.sel(DATE = low_mid_idxs) 

#z @ Peuil de Claix: 935 m ; z @ pont de claix: 237 m ; dz in km
dz = (935 - 237)/1000

for year in np.unique(pd.DatetimeIndex(low_mid_idxs).year):
    
    
    #t1 = np.logical_and(chartreuse['time.year']==year, chartreuse['time.month']>=10)
    t1 = np.logical_and(data_high['DATE.year']==year, data_high['DATE.month']>=10)
    t2 = np.logical_and(data_high['DATE.year']==year+1, data_high['DATE.month']<=4)
    t = data_high.DATE[np.logical_or(t1,t2)]
    
    t_high = data_high.sel(DATE = t).values
    t_high = t_high.reshape(t_high.size,1)
    t_low = data_low.sel(DATE = t).values.reshape(t_high.shape)
    
    dzt = (t_high - t_low)/dz
    
    #InSitu gradient - Low Mid (ISLM)
    filename = f'ISLM_{year}.npy'
    filepath = os.path.join(savepath,filename)
    
    np.save(filepath, dzt)


#mid_high dzt loop:
data_high = cdp_dailyT.sel(time = mid_high_idxs) - 273.15
data_low = mid_values.T.sel(DATE = mid_high_idxs)

#z @ col de porte: 1325 m ; z @ peuil de claix: 935 m ; dz in km
dz = (1325 - 935)/1000

for year in np.unique(pd.DatetimeIndex(mid_high_idxs).year):
    
    
    #t1 = np.logical_and(chartreuse['time.year']==year, chartreuse['time.month']>=10)
    t1 = np.logical_and(data_high['time.year']==year, data_high['time.month']>=10)
    t2 = np.logical_and(data_high['time.year']==year+1, data_high['time.month']<=4)
    t = data_high.time[np.logical_or(t1,t2)]
    
    t_high = data_high.sel(time = t).values
    t_low = data_low.sel(DATE = t).values.reshape(t_high.shape)
    
    dzt = (t_high - t_low)/dz
    
    #InSitu gradient - Mid High (ISMH)
    filename = f'ISMH_{year}.npy'
    filepath = os.path.join(savepath,filename)
    
    np.save(filepath, dzt)


#%% Normalization of generated files

def normalize_data(label: str = 'ISLH',
                   data_path: str = './NN_data/'):
    
    #walk through directory to find years and normalize
    data_min = 1000.
    data_max = -1000.
    
    #read loop to find global min and max
    for root, directories, files in os.walk(data_path):
        for filename in files:
            if filename[-4:] == '.npy':
                if filename[:4] == label:
                    path = root + filename
                    data = np.load(path)
                    data_min = min(data_min, data.min())
                    data_max = max(data_max, data.max())
    
    #saving info needed to reconstruct data
    reconstructor = {
        'min': data_min,
        'range': data_max - data_min
        }
    recon_filename = f'{label}_reconstructor.pkl'
    recon_path = os.path.join(data_path, recon_filename)
    
    with open(recon_path, 'wb') as handle:
        pickle.dump(reconstructor, handle)
    
    #read and write loop to normalize
    for root, directories, files in os.walk(data_path):
        for filename in files:
            if filename[-4:] == '.npy':
                if filename[:4] == label:
                    path = root + filename
                    data = np.load(path)
                    data = (data - data_min)/(data_max-data_min)
                    np.save(path,data)

labels = ['ISLH', 'ISLM', 'ISMH']
for var in labels: normalize_data(label=var)     
    



#%% Plots for meeting

ylim = (-10,40)
xlim =( heal_data['ILL_T2'].asfreq('H')['2013'].dropna().index[0],heal_data['ILL_T2'].asfreq('H')['2013'].dropna().index[-1])

pont['2013'].plot(linewidth = 0.1, marker = 'o', ylim=ylim, title='Pont 2013')

test['Pont_de_Claix']['2013'].plot(linewidth = 0.1, marker = 'o', ylim=ylim, title='Pont 2013 raw', c='orange')
heal_data['ILL_T2'].dropna()['2013'].plot(linewidth = 0.1, marker = 'o', ylim=ylim, xlim=xlim, title='ILL_T2 raw', c='cyan')
tstplot = low_regressor.predict(heal_data['ILL_T2'].asfreq('H')['2013'].dropna())

fig, ax = plt.subplots(1,1)
plt.ylim(ylim)
fig.suptitle('ILL Regresion 2013')
plt.xlim(xlim)
ax.plot(heal_data['ILL_T2'].asfreq('H')['2013'].dropna().index, tstplot, linewidth = 0.1, marker = 'o', c='r')



