This readme describes the folders and scripts in the 21AirPolML project. Some
data may not be made available on the repository until approval is given from
the parties responsible for, e.g., the input data.

Where possible, sample data will be given so function can be demonstrated.

-Milton S. Gomez, July 22nd, 2021


#Folders                                                                    
                                                                            
##NN_data 
>includes numpy arrays and pickle files for input and target data used when training/applying neural network. See folder readme for details.

##NN_results 
>includes predictions from the Neural Network using ERA5 and MPI based pressure fields. See folder readme for details. 

##obs_raw
>includes raw data for observations from weather stations 

##saves 
>includes .h5 files for saved keras models

##logs
>includes logs from neural network training callbacks                 │

#Scripts                                                                    │
│                                                                            │
│data_gen.py - data generator scripts for indexing and creating the data     │
│    generator for training the neural network.                              │
│                                                                            │
│MAR_Pressure.py - Script used to compare MAR pressure outputs when forced by│
│ERA5 vs forced by MPI. Also used to export pressure files used to train or  │
│apply network.                                                              │
│                                                                            │
│nn_main_script - script that trains the neural network.                     │
│                                                                            │
│nn_prediction_script.py - script for generating predictions from the trained│
│    neural network.                                                         │
│                                                                            │
│nn_result_comparison.py - script for generating plots to compare neural     │
│    network predictions.                                                    │
│                                                                            │
│observation_read.py - script for reading observation files and generating   │
│    the files containing the target temperature gradient.                   │
│                                                                            │
│Tensorboard.ipnyb - jupyter notebook for loading a tensorboard instance to  │
│    monitor neural network during/after training.                           │
│                                                                            │
╰────────────────────────────────────────────────────────────────────────────╯
