#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 09:31:02 2021

This script applies a saved model to a set of input fields. Script originally
compared predictions to target values, but functionality has been deprecated.

@author: Milton Gomez
"""

import numpy as np
import pickle
import tensorflow.keras as keras
import matplotlib.pyplot as plt

"""──────────────────────────────────────────────────────────────────────────┐
│The following variables point to locations where key components are stored. │
│                                                                            │
│filepath points to the directory where the keras models are saved (.h5      │
│files). Note that this folder also contains the saved parameter dictionaries│
│should there be a need to inspect them.                                     │
│                                                                            │
│data_path points to the NN_data path; this was used in the deprecated       │
│functionality mentioned above.                                              │
└──────────────────────────────────────────────────────────────────────────"""


data_path = './NN_data/'
filepath = './saves/'

model_name = '2021.05.20-10:44:45' #Model trained for use in manuscript: 2021.05.20-10:44:45

model_path = f'{filepath}model_{model_name}.h5'
param_path = f'{filepath}params_{model_name}.pkl'

save_name = 'ERAnorm'

#loading the model and the hyperparameters used
model = keras.models.load_model(model_path)
with open(param_path,'rb') as handle:
    params = pickle.load(handle)

#shape of the time series image input
#[batch_size, num_days, y_length, x_length, channels]
input_ts_shape = model.input_shape[0]


#year to be observed
year_list = np.arange(1981,1982,1)
for year in year_list:
    #loading input data
    in_data = []
    for var in params['in_vars']:
        in_data.append(np.load(f'{data_path}SSEN_{year}.npy'))
    in_data = np.array(in_data)
    in_data = np.moveaxis(in_data,0,-1)
    
    """───────────────────────────────────────────────────────────────────────────
    │   deprecated code   │ 
    └─────────────────────┘
    
    #loading the target values
    #out_data = []
    #for var in params['out_vars']:
    #    out_data.append(np.load(f'{data_path}{var}_{year}.npy'))
    #out_data = np.array(out_data).flatten() #outout is 1D, flatten for simplicity
    └──────────────────────────────────────────────────────────────────────────"""
    
    predict = []
    
    #i to iterate through batches
    for i in range( int( in_data.shape[0] / input_ts_shape[0]) +1):
        print(i)
        step = params['in_step']
        batch_size = input_ts_shape[0]
        day_idxs = (i*batch_size + np.arange(batch_size).reshape(batch_size)) / (212 - step)
        depth = np.sin(np.pi * day_idxs)
        
        data_batch = []
        #j to iterate through each member of the batch
        for j in range(batch_size):
            start_idx = j + i*batch_size
            if ((start_idx + step) <= in_data.shape[0]):
                data_batch.append(in_data[start_idx:start_idx + step])
            else:
                data_batch.append(np.zeros((step, *in_data.shape[1:])))
                
        data_batch = np.array(data_batch)
         
        predict.append(
            model.predict([data_batch, depth])
            )
    predict = np.array(predict).flatten()[:in_data.shape[0]-(step-1)]
    
    #plotting parameters
    ylim=[0,1]
    
    #plotting results
    fig, axs = plt.subplots(nrows=1,ncols=1)
    fig.suptitle(f'Predicted normalized temperature gradient for {year}-{year+1} winter')
    
    """───────────────────────────────────────────────────────────────────────────
    │   deprecated code   │ 
    └─────────────────────┘
    #axs.plot(out_data[(step-1):], color='blue')
    #axs.axhline(out_data[~np.isnan(out_data)].mean(), color='cyan')
    #axs[1].axhline(out_data[~np.isnan(out_data)].mean(), color='cyan')
    #axs[1].set_ylim(ylim)
    
    #check how many days in the time series are over the mean for the year
    #inv_actual = (out_data>out_data[~np.isnan(out_data)].mean()).sum()
    
    #print(f'{year} Actual days over mean dzt: {inv_actual}')
    print(f'{year} Days predicted to be over predicted mean dzt: {inv_hat}')
    #print(f'{year} relative accuracy: {(inv_hat - inv_actual)/inv_actual} ')
    └──────────────────────────────────────────────────────────────────────────"""
    
    axs.set_ylim(ylim)
    axs.plot(predict, color='red')
    axs.axhline(predict.mean(), color='pink')
    
    inv_hat = (predict.flatten()>predict.mean()).sum()
    
    predict_path = f'./NN_results/dzt_{save_name}_{year}.npy'
    np.save(predict_path, predict)
    

