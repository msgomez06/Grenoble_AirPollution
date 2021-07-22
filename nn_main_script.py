# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 14:05:19 2021

Neural Network Script - Builds, trains, and saves Keras model.

@author: Milton Gomez
"""

"""──────────────────────────────────────────────────────────────────────────┐
│ Loading necessary libraries to build and train model                       │
└──────────────────────────────────────────────────────────────────────────"""
import numpy as np
import pickle
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import TimeDistributed as td
import data_gen as dg #Data Generator built for this model; check data_gen.py


"""──────────────────────────────────────────────────────────────────────────┐
│Libraries needed for timestamping model and changing TF verbosity so        │
│running interactively is not overwhelming.                                  │
└──────────────────────────────────────────────────────────────────────────"""
from datetime import datetime
import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.WARNING)
#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARNING)



"""──────────────────────────────────────────────────────────────────────────┐
│Flag for training model. Sometimes we just want to check for proper building│
└──────────────────────────────────────────────────────────────────────────"""
run = False


"""──────────────────────────────────────────────────────────────────────────┐
│Available data variables stored in NN_data folder. Please review the readme │
│file to check for variable names and other details.                         │
└──────────────────────────────────────────────────────────────────────────"""

"""──────────────────────────────────────────────────────────────────────────┐
│Hyperparameters for the neural network are defined in this section. Note    │
│that the hyperparameters will be stored as a dictionary upon successful     │
│training of the model.                                                      │
└──────────────────────────────────────────────────────────────────────────"""

#Hyperparameters governing Training regimen
###primary training loop. Default LR schedule has been defined as exp. decay
num_epochs = 400 #Number of epochs for the first training loop
etta_0 = 2e-4 #default learning rate
decay_rate = 5e-2 #see decay_epochs for place in formula
decay_epochs = int(num_epochs*1) #num of epochs until etta = etta_0 * decay_rate

###secondary training loops. Idea is to perform tuning at small, constant LR.
mid_tune_epochs = 30 #epochs for secondary loop tuning
mid_tune_etta = 1e-5 #constant LR used in secondary loop

###fine tuning training loop. VGG19 layers unfrozen for these epochs
tuning_epochs = 50 #epochs for fine-tuning. Keep low since very comp. expensive
tuning_decay = 4e-6 #small, constant LR used in tuning. Large LR => overfitting

###general training parameters
batch_size = 64 #batch size for training
shuffle = True #shuffling between epochs

#Hyperparameters governing inputs/outputs
###Vars to be used with NN. Currently architecture expects single var for input
in_vars =  ['SLPN'] #list of invar names
out_vars = ['ISLH'] #list of outvar names
in_step = 4 #number of days to consider for RNN time series input
out_step = 1 #number of days for RNN output. =1 unless RNN is many-to-many

#Hyperparameters governing dataset splitting
test_size = 5 #number of years to consider as test data. Uses n most recent yrs
split = 0.2 #validation set adds years until (val. size/train size) > split

#Misc Hyperparameters
data_path = './NN_data/' #path to NN data
random_seed = None #set to None if not using. Sets random state.
lstm_units= 1024 #number of units in the LSTM layer of the architecture
notes = f'vgg19_manuscript2'#callback string; differentiate runs tensorboard

#setting up parameter dictionary for saving
global_params = {
            'num_epochs': num_epochs,
            'in_vars':  in_vars,
            'out_vars':  out_vars,
            'in_step': in_step,
            'out_step': out_step,
            'split': split, 
            'test_size': test_size,
            'batch_size' : batch_size,
            'shuffle' : shuffle,
            'data_path': data_path,
            'random_seed' : random_seed,
            'etta_0' : etta_0,
            'decay_epochs' : decay_epochs,
            'decay_rate' : decay_rate,
            'lstm_units' : lstm_units,
            'mid_tune_epochs' : mid_tune_epochs,
            'mid_tune_etta' : mid_tune_etta
            }

#setting up random number generator:
if random_seed != None: 
    tf.random.set_seed(random_seed)
    randomizer = np.random.default_rng(random_seed)
else:
    randomizer = np.random.default_rng()


#setting up indexer parameters
idxr_params = {
            'in_vars':  in_vars,
            'out_vars':  out_vars,
            'in_step': in_step,
            'out_step': out_step,
            'split': split, 
            'test_size': test_size,
            'data_path': data_path,
            'randomizer': randomizer
            }

#setting up data generator parameters
dg_params = { 
            'in_vars': in_vars,
            'out_vars': out_vars, 
            'in_step': in_step,
            'out_step': out_step,
            'batch_size': batch_size, 
            'shuffle': shuffle,
            'data_path': data_path,
            'randomizer': randomizer
            }

# Indexing to find valid IDs
ids = dg.indexer(**idxr_params)

#%% Tensorflow Distributed Strategy and Hyperparameters

"""──────────────────────────────────────────────────────────────────────────┐
│Neural network developed to work on GPU cluster at LEGI laboratory. Maximum │
│of 4 GPU cores available at time of writing. Due to resource sharing, most  │
│trials were run with 2 of the 4 GPUs.                                       │
└──────────────────────────────────────────────────────────────────────────"""
#set up multiple GPU strategy
physical_devices = ['/gpu:2', '/gpu:3'] #list of GPUs to use
strategy = tf.distribute.MirroredStrategy(devices=physical_devices)
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
tf.config.experimental.set_synchronous_execution(enable=True)

"""──────────────────────────────────────────────────────────────────────────┐
│For the final run, both training and validation datasets were iterated      │
│through to train the network. If training and validation are needed for     │
│experimentation, create two generators and pass them into fit function      │
└──────────────────────────────────────────────────────────────────────────"""
# Generator
data_generator = dg.DataGenerator_V2(ids['train'] + ids['validation'],
                                     **dg_params)

"""──────────────────────────────────────────────────────────────────────────┐
│Defining the learning rate schedule. Note that the default is exponential   │
│decay.                                                                      │
└──────────────────────────────────────────────────────────────────────────"""
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate = etta_0,
    decay_steps = decay_epochs,
    decay_rate = decay_rate)

"""──────────────────────────────────────────────────────────────────────────┐
│Optimizer definition. Note that the learning rate decay is applied in the   │
│callbacks and not in the optimizer. Beta vals left at default, but epsilon  │
│was set to two since it was found to work well during experimentation       │
└──────────────────────────────────────────────────────────────────────────"""
opt = tf.keras.optimizers.Adam(
    learning_rate=etta_0,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-2 )

#%%Model Definition

"""──────────────────────────────────────────────────────────────────────────┐
│Functions for using transfer learning with VGG19. build_vgg uses the full   │
│16 non-dense layers (i.e., only uses the convolutional feature extractor    │
│from VGG19). build_vgg_sub allows using a portion of the feature extractor, │
│and was found to be more memory intensive due to the number of parameters.  │
└──────────────────────────────────────────────────────────────────────────"""
def build_vgg(in_shape):
    vgg = keras.applications.VGG19(
        include_top=False,
        weights='imagenet',
        input_shape = in_shape
        )
    return vgg

def build_vgg_sub(in_shape):
    vgg = keras.applications.VGG19(
        include_top=False,
        weights='imagenet',
        input_shape = in_shape
        )
    outputs = vgg.layers[-6].output
    
    return keras.Model(vgg.input,outputs)


"""──────────────────────────────────────────────────────────────────────────┐
│Defining the model within the mirrored strategy. Note that this will cut the│
│batch into portions equally distributed amongst GPU cores, thus batch size  │
│must be evenly divisible by the number of cores.                            │
└──────────────────────────────────────────────────────────────────────────"""
with strategy.scope():

    """──────────────────────────────────────────────────────────────────────┐
    │Input returns a field variable time series and a single representation  │
    │of the depth into the winter. The field input is written as SP_input    │
    │due to historical nomenclature but should be updated to reflect it is   │
    │the defined invar and not the surface pressure.                         │
    │Field input needs to be reformatted to be a 3-channel 'image' with 8-bit│
    │channels, as this is what VGG19 expects.                                │
    └──────────────────────────────────────────────────────────────────────"""    
    SP_input = keras.Input((*data_generator.dim,data_generator.n_channels),data_generator.batch_size)
    depth_input = keras.Input(1, batch_size=data_generator.batch_size)
    
    scaled_SP = keras.layers.experimental.preprocessing.Rescaling(255)(SP_input)
    
    scaled_input = keras.layers.Concatenate()([scaled_SP,scaled_SP,scaled_SP])
    
    processed_input = keras.applications.vgg19.preprocess_input(scaled_input)
    
    
    """──────────────────────────────────────────────────────────────────────┐
    │Data augmentation. Though some papers indicate gaussian noise may not be│
    │ideal, it was used during these experiments. A rotation augmentation was│
    │also applied.                                                           │
    └──────────────────────────────────────────────────────────────────────"""
    rotation_augment = td(keras.layers.experimental.preprocessing.RandomRotation(.01, fill_mode='nearest'))(processed_input)
    noise_augment = td(keras.layers.GaussianNoise(1))(rotation_augment)
    
    VGG = build_vgg(processed_input.shape[2:])
    VGG.trainable=False #freeze VGG19 layers for first and second train rounds
    
    hidden = td(VGG)(noise_augment)
    drop2d = td(keras.layers.SpatialDropout2D(0.2))(hidden)
    flat = td(keras.layers.Flatten())(drop2d)
    
    #recurrent section
    lstm = keras.layers.LSTM(units=lstm_units)(flat)
    
    
    #Concatenating LSTM units with winter depth input
    concatenate = keras.layers.Concatenate()([lstm,depth_input])
    
    drop0 = keras.layers.Dropout(0.10)(concatenate)
    
    dense1 = keras.layers.Dense(512, activation='relu')(drop0)
    
    drop1 = keras.layers.Dropout(0.10)(dense1)
    
    dense2 = keras.layers.Dense(256, activation='relu')(drop1)
    
    drop2 = keras.layers.Dropout(0.10)(dense2)
    
    dense3 = keras.layers.Dense(256, activation='relu')(drop2)
    
    drop3 = keras.layers.Dropout(0.08)(dense3)
    
    #ouput layer built to match out_data shape.
    dense4 = keras.layers.Dense(data_generator.out_step*len(data_generator.out_vars), activation='relu')(drop3)
    
    #MSE as loss due to this being a regression problem
    loss_fn = keras.losses.MeanSquaredError(reduction='none')
    
    #metrics to get idea of performance. MAPE particularly useful.
    metrics = [keras.metrics.MeanAbsolutePercentageError(), 
               keras.metrics.RootMeanSquaredError()]
    
    model = keras.models.Model([SP_input, depth_input], dense4)
    model.compile(loss=loss_fn, metrics=metrics, optimizer=opt)
#------end of strategy section---------#

"""──────────────────────────────────────────────────────────────────────────┐
│The timestamp is going to be used to both identify the model when it is     │
│saved and to complement the notes when making the log dir associated with   │
│the run (this allows differentiation between runs that have the same        │
│hyperparameters).                                                           │
│                                                                            │
│logdir includes timestamp, notes, input varnames, output varnames           │
└──────────────────────────────────────────────────────────────────────────"""
time_str = datetime.now().strftime("%Y.%m.%d-%H:%M:%S")
log_dir = os.path.join(os.path.curdir, 'logs', time_str, notes) #+ os.path.sep
log_dir += '_'
for varname in in_vars: log_dir += '.' + varname
log_dir += '_'
for varname in out_vars: log_dir += '.' + varname

tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir)
learning_callback = keras.callbacks.LearningRateScheduler(lr_schedule, verbose=1)


callbacks = keras.callbacks.CallbackList()
callbacks.append(tensorboard_callback)
callbacks.append(learning_callback)


#%% Training Section

if __name__  == "__main__" and run:
    # Train model on dataset
    #main training loop ; has exponential decay rate
    model.fit(data_generator,
                        #validation_data=validation_generator, #removed since not validating
                        workers=1, epochs=num_epochs, verbose=0, callbacks=callbacks)
    
    #second fine tune learning rate
    learning_callback.schedule = lambda step: mid_tune_etta
    model.compile(loss=loss_fn, metrics=metrics, optimizer=keras.optimizers.SGD(learning_rate = 1e-6, momentum=0.9))
    
    model.fit(data_generator,
                        #validation_data=validation_generator, #removed since not validating
                        initial_epoch=num_epochs,
                        workers=1, epochs=num_epochs + mid_tune_epochs, verbose=0, callbacks=callbacks)
    
    model.trainable=True
    learning_callback.schedule = lambda step: tuning_decay
    model.compile(loss=loss_fn, metrics=metrics, optimizer=keras.optimizers.Adam(learning_rate = 1e-6, epsilon=1e-2))
    print('Tuning')
    model.fit(data_generator,
                        #validation_data=validation_generator, #removed since not validating
                        initial_epoch=num_epochs + mid_tune_epochs + 1,
                        workers=1, epochs=num_epochs + mid_tune_epochs + tuning_epochs, verbose=0, callbacks=callbacks)
    
    #save model and parameters
    keras.models.save_model(model, os.path.join(os.path.curdir, 'saves',f'model_{time_str}.h5'))
    with open(os.path.join(os.path.curdir, 'saves',f'params_{time_str}.pkl'),'wb') as handle:
        pickle.dump(global_params, handle)


