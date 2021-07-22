# -*- coding: utf-8 -*-
"""
This script is used to evaluate how well the neural network predicts the 
temperature gradient. Two types of comparisons can be made: between the 
predictions using ERA5 and the observations, or between the predictions
using ERA5 and the predictions using MPI.

@author: Milton Gomez
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import xarray as xr
import pickle
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator

import scipy.stats as stats

"""──────────────────────────────────────────────────────────────────────────┐
│This script was developed to make comparisons using the predictions from the│
│neural network.                                                             │
│                                                                            │
│Observation flag sets whether the comparisons will be made between ERA5 and │
│observations (True) or between ERA5 and MPI (False)                         │
│                                                                            │
│Color scheme for plots: observations blue, ERA5->MAR red, MPI->MAR green.   │
│                                                                            │
│Please forgive the general messiness, as this was meant as a run-when-needed│
│modify-for-current-purpose script.                                          │
│                                                                            │
│source refers to variable that predictions will be compared against,        │
│whether observations or predictions from NN using MAR forced by ERA5        │
│pressure fields.                                                            │
│                                                                            │
│target refers to predicted variable that will be compared against source    │
└──────────────────────────────────────────────────────────────────────────"""
observation = True

if observation:
    source_path =  './NN_data/'    
    source_var = 'ISLH'
    target_var = 'dzt_hat'
    
    src_c1 = 'blue'
    src_c2 = 'cyan'
    tgt_c1 = 'red'
    tgt_c2 = 'orange'
    
else:
    source_path =  './NN_results/'   
    source_var = 'dzt_hat'    
    target_var = 'dzt_ERAnorm'
    
    src_c1 = 'red'
    src_c2 = 'orange'
    tgt_c1 = 'green'
    tgt_c2 = 'chartreuse'

"""──────────────────────────────────────────────────────────────────────────┐
│Loading the reconstructor. Needed to translate normed values to real values.│
│Since network was trained to predict ISLH var, all predictions reconstructed│
│using ISLH reconstructor.                                                   │
└──────────────────────────────────────────────────────────────────────────"""
recon_path = './NN_data/'
with open(f'{recon_path}ISLH_reconstructor.pkl','rb') as handle:
    recon = pickle.load(handle)

#path to the predictions folder
results_path = './NN_results/'

#initialize var where xarray dataset will be stored
dzt = None

"""──────────────────────────────────────────────────────────────────────────┐
│Defining the time period for the comparison.                                │
│                                                                            │
│Max available observations: 1993 - 2017 (winters), 1993 start, 25 years     │
│                                                                            │
│Available ERA5 data: 1980-2017                                              │
│Available MPI data: 1981-2100                                               │
└──────────────────────────────────────────────────────────────────────────"""
start_year = 1993
num_years = 25
year_list = np.arange(start_year, start_year + num_years, 1)


#Building the x_array for simpler time handling
for year in year_list:
    print(f'Processing {year}')
    
    #Build time index for keeping track of date.
    #since 4-day time series was used, wintertime is from Oct-04 to APR-30
    date_index = np.arange(np.datetime64(f'{year}-10-04'),#date start
                       np.datetime64(f'{year+1}-05-01'),#date_stop (not inclusive)
                       np.timedelta64(1,'D')) #date_step, 1 day
    
    #set up a winter ID for easier filtering by winter period
    winter_idx = np.multiply(np.ones_like(date_index).astype('int'), year)
    
    x = np.load(f'{source_path}{source_var}_{year}.npy')*recon['range']+recon['min']
    x_hat = np.load(f'{results_path}{target_var}_{year}.npy')*recon['range']+recon['min']
    
    if observation:
        #observations have the 1st 3 days of the winter, which have no nn prediction
        source_data = xr.DataArray(x[3:].flatten(), coords={'time':date_index}, dims=['time'], name='source')
    else:
        source_data = xr.DataArray(x.flatten(), coords={'time':date_index}, dims=['time'], name='source')
    #hat is the prediction to compare (target variable). 
    hat = xr.DataArray(x_hat.flatten(), coords={'time':date_index}, dims=['time'], name='target')
    wint = xr.DataArray(winter_idx, coords={'time':date_index}, dims=['time'], name='winter')
    
    temp = xr.Dataset({source_data.name:source_data,
                       hat.name:hat, 
                       wint.name:wint})    
    #temp['Observed'].plot()

    if year == start_year:
        dzt = temp.copy()    
    else:
        dzt = xr.concat((dzt,temp), dim='time')

#filtering out nan values in source        
source = dzt['source'].where(~np.isnan(dzt['source']), drop=True)
hat = dzt['target']

#using scipy to find kernel density estimator using scotts mean. used for PDFs
source_kernel = stats.kde.gaussian_kde(source)
predicted_kernel = stats.kde.gaussian_kde(hat)

#values for making pdfs. These were found empirically.
x_vals = np.arange(-12.5,12.5,.1)


#%% Probability Density Function - Separate
"""──────────────────────────────────────────────────────────────────────────┐
│This section generates two PDFs to be plotted in a two row figure.          │
│                                                                            │
│Comparison with visible normalized histogram bars .                         │
└──────────────────────────────────────────────────────────────────────────"""
title = f'dzT PDF over {num_years} year Period, starting {start_year}'
ylim = [0,0.325]
size = (8,6)
dpi=150

if observation:
    source_desc = 'Observations from Pont de Claix and Col de Porte'
    target_desc = 'NN Predictions from ERA5->MAR Inputs'
    

else:
    source_desc = 'NN Predictions from ERA5->MAR Inputs'
    target_desc = 'NN Predictions from MPI->MAR Inputs'
    

    
fig, axs = plt.subplots(2,1,sharex=True, sharey=True, figsize=size, tight_layout=True, dpi=dpi)
fig.suptitle(title)
axs[0].set_ylim(ylim)


axs[0].plot(x_vals, source_kernel(x_vals), label='PDF', color=src_c2)
axs[1].plot(x_vals, predicted_kernel(x_vals), label='PDF', color=tgt_c2)
axs[0].hist(source, bins=np.arange(-12.5,12.5,1), density=True, color=src_c1, label='Histogram')
axs[1].hist(hat, bins=np.arange(-12.5,12.5,1), density=True, color=tgt_c1, label='Histogram')
handles0, labels0 = axs[0].get_legend_handles_labels()
handles1, labels1 = axs[1].get_legend_handles_labels()

axs[0].title.set_text(f'$d_zT$ {source_desc}')
axs[1].title.set_text(f'$d_zT$ Predicted using {target_desc}')

#Adding notes to legend with statistics
###Notes for source
note0_0 = mpatches.Patch(color=None, alpha=0, label= f'Median: {float(source.median()):.2f}')
note0_1 = mpatches.Patch(color=None, alpha=0, label= '$5^{th}$ Percentile: ' + f'{float(np.percentile(source,5)):.2f}')
note0_2 = mpatches.Patch(color=None, alpha=0, label= '$95^{th}$ Percentile: ' + f'{float(np.percentile(source,95)):.2f}')
note0_3 = mpatches.Patch(color=None, alpha=0, label= f'Std dev: {float(source.std()):.2f}')
for i in range(4): handles0.append(eval(f'note0_{i}'))

###Notes for target
note1_0 = mpatches.Patch(color=None, alpha=0, label= f'Median: {float(hat.median()):.2f}')
note1_1 = mpatches.Patch(color=None, alpha=0, label= '$5^{th}$ Percentile: ' + f'{float(np.percentile(hat,5)):.2f}')
note1_2 = mpatches.Patch(color=None, alpha=0, label= '$95^{th}$ Percentile: ' + f'{float(np.percentile(hat,95)):.2f}')
note1_3 = mpatches.Patch(color=None, alpha=0, label= f'Std dev: {float(hat.std()):.2f}')
for i in range(4): handles1.append(eval(f'note1_{i}'))

axs[0].legend(handles = handles0, loc='upper right')
axs[1].legend(handles = handles1, loc='upper right')

#%%PDF Overlapped
"""──────────────────────────────────────────────────────────────────────────┐
│This section generates two PDFs to be plotted in a single figure, with the  │
│source in blue and predictions in red.                                      │
│Good overlapping comparison                                                 │
└──────────────────────────────────────────────────────────────────────────"""
if observation:
    source_desc = 'Observations'
    target_desc = 'NN Predictions (ERA5->MAR)'
    
else:
    source_desc = 'NN Predictions (ERA5->MAR)'
    target_desc = 'NN Predictions (MPI->MAR)'


fig0, ax = plt.subplots(1,1, figsize=(8,3), dpi=200)
#fig2.suptitle(f'Q-Q Plot {start_year}-{start_year + num_years - 1 }')
ylim = [0,0.325]

#ax.set_xlim([-12.5,12.5])
ax.set_ylim(ylim)

#plotting
ax.plot(x_vals, source_kernel(x_vals), label=source_desc, color=src_c1)
ax.plot(x_vals, predicted_kernel(x_vals), label=target_desc, color=tgt_c1)
#ax.xaxis.label.set_text('MPI NN Prediction Quantiles')

ax.autoscale(enable=True, axis='x', tight=True)

ax.xaxis.label.set_text('Temperature Gradient')
ax.yaxis.label.set_text('Normalized Probability')
ax.legend(loc='upper right')

fig0.tight_layout()

#%%Quantile Plot
"""──────────────────────────────────────────────────────────────────────────┐
│This section of the script plots a quantile-quantile plot for the source    │
│and target. Note that the quantiles plotted are the 1st percentile to 99th  │
│percentile with 1 percent increments.                                       │
└──────────────────────────────────────────────────────────────────────────"""
q_array = np.arange(.01,.99,.01)
source_quantile = np.quantile(source,q_array)
hat_quantile = np.quantile(hat, q_array)

if observation:
    source_desc = 'Observation Values (K/km)'
    target_desc = 'NN Prediction Values (K/km) ERA5->MAR'
else:
    source_desc = 'NN Prediction Values (K/km) ERA5->MAR'
    target_desc = 'NN Prediction Values (K/km) MPI->MAR'

fig2, ax = plt.subplots(1,1, figsize=(4.5,4.5), dpi=200)
#fig2.suptitle(f'Q-Q Plot {start_year}-{start_year + num_years - 1 }')

ax.set_aspect(1)
ticks = np.arange(-10,5,2)
ax.xaxis.set_ticks(ticks)
ax.yaxis.set_ticks(ticks)

ax.set_xlim([-10,4])
ax.set_ylim([-10,4])
ax.plot([0, 1], [0, 1], transform=ax.transAxes, color='black', linewidth=0.5)
ax.scatter(source_quantile, hat_quantile, s=5, c=q_array, cmap='winter')

ax.xaxis.label.set_text(f'{source_desc}')
ax.yaxis.label.set_text(f'{target_desc}')
fig2.tight_layout()

#%%Correlation Plot
"""──────────────────────────────────────────────────────────────────────────┐
│This section plots the correlation between the source and the target. Labels│
│include mean absolute error and pearson correlation coefficient.            │
└──────────────────────────────────────────────────────────────────────────"""
correlation = np.corrcoef(source,hat.sel(time=source['time']),rowvar=False)[0,1]
MAE = np.abs(source-hat.sel(time=source['time'])).sum()/source.size

desc_text = f'Correlation Coefficient: {correlation:.2f}' 
desc_text +=f'\n  Mean Absolute Error: {float(MAE):.2f}'

if observation:
    x_text = 'Observation (K/km)'
    y_text = 'NN Prediction (K/km) ERA5->MAR'
else:
    x_text = 'NN Prediction (K/km) ERA5->MAR'
    y_text = 'NN Prediction (K/km) MPI->MAR'

ticks = np.arange(-12,12,4)

fig3, ax = plt.subplots(1,1, figsize=(4.5,4.5), dpi=200)
#fig3.suptitle(f'Correlation Plot {start_year}-{start_year + num_years - 1 } {desc_text}')
fig3.suptitle(desc_text)

ax.set_aspect(1)

ax.set_xlim([-12,8])
ax.set_ylim([-12,8])
ax.xaxis.set_ticks(ticks)
ax.yaxis.set_ticks(ticks)
ax.plot([0, 1], [0, 1], transform=ax.transAxes, color='black', linewidth=0.5)
ax.scatter(source, hat.sel(time=source['time']), s=1.5, c='steelblue')

ax.xaxis.label.set_text(x_text)
ax.yaxis.label.set_text(y_text)

fig3.tight_layout()

#%%time series plot
"""──────────────────────────────────────────────────────────────────────────┐
│The time series plot is only valid when comparing observations to ERA5->MAR,│
│as ERA5 and MPI are not time-correspondent. Months restricted to Nov-Mar to │
│match periods studied by Largeron and Staquet, 2016.                        │
└──────────────────────────────────────────────────────────────────────────"""
if observation:
    
    #choosing the year for the visualization
    sel_yr = 2015

    data = dzt.where(dzt['winter']==sel_yr, drop=True)
    data = data.where(np.logical_or(data['time.month']>=11,data['time.month']<=3), drop=True)
    
    data_2 = data.where(~np.isnan(data['source']),drop=True)
    
    fig3, ax = plt.subplots(1,1, figsize=(6,3), dpi=200)
    ax.xaxis.label.set_text('Date')
    
    
    fmt_month = mdates.MonthLocator()
    ax.xaxis.set_minor_locator(fmt_month)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    
    #ax.autoscale(enable=True, axis='x', tight=True)
    ax.yaxis.label.set_text('Temperature Gradient (K/km)')

    ax.plot(data['time'], data['source'], label="Observations", color=src_c1)
    ax.plot(data['time'], data['target'], label="NN Prediction", color=tgt_c1)
    ax.legend()
    
    fig3.autofmt_xdate()
    fig3.tight_layout(pad=1.25)

#%% Median Series Plot
"""──────────────────────────────────────────────────────────────────────────┐
│Plots to inspect the evolution of the yearly median throughout the time     │
│series. This isn't implemented with resampling due to using winter seasons  │
└──────────────────────────────────────────────────────────────────────────"""

if observation:
    src_label = 'Median Observation (K/km)'
    tgt_label = 'Median NN Prediction (K/km) ERA5->MAR'
else:
    src_label = 'Median NN Prediction (K/km) ERA5->MAR'
    tgt_label = 'Median NN Prediction (K/km) MPI->MAR'

src_median = np.empty((0),dtype='float')
tgt_median = np.empty((0),dtype='float')
year_iter = np.unique(dzt['winter'])
for year in year_iter:
    temp = dzt.where(dzt['winter']==year, drop=True)
    src_median = np.append(src_median,temp['source'].median().values)
    tgt_median = np.append(tgt_median,temp['target'].median().values)

#median series plot
fig3, ax = plt.subplots(1,1, figsize=(8,2.25), dpi=200)
ax.xaxis.label.set_text('Year')
ax.xaxis.set_minor_locator(MultipleLocator(1))
ax.autoscale(enable=True, axis='x', tight=True)
ax.yaxis.label.set_text('Median Temperature \n Gradient (K/km)')
ax.scatter(year_iter, src_median, label=src_label, color=src_c1, s=6)
ax.scatter(year_iter, tgt_median, label=tgt_label, color=tgt_c1, s=4)

"""──────────────────────────────────────────────────────────────────────────┐
│linear regressions were found to have a pvalue > .10 and thus not plotted   │
│since not significant. Code left here as reference should it be needed.     │
└──────────────────────────────────────────────────────────────────────────"""
##linear_regressions
#src_reg = stats.linregress(year_iter, src_median)
#tgt_reg = stats.linregress(year_iter, tgt_median)
#src_trend = src_reg.intercept + src_reg.slope * year_iter
#tgt_trend = tgt_reg.intercept + tgt_reg.slope * year_iter

#ax.plot(year_iter,src_trend,color=src_c2)
#ax.plot(year_iter,tgt_trend,color=tgt_c2)
ax.grid(axis='x', which='both')
ax.legend()


        
