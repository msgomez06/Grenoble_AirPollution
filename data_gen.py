# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 15:42:21 2021

Data generator based on article published on:
    https://web.archive.org/web/20201204052832/https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

@author: Milton Gomez
"""
import numpy as np
import tensorflow.keras as keras
import os


"""──────────────────────────────────────────────────────────────────────────┐
│DataGenerator  is the legacy data generator configured to only handle       │
│surface pressure as the input. The final generator used is DataGeneratorV2. │
│DataGenerator is included for historical references, but can be deleted     │
│without affecting the functioning of the scripts included in the folder.    │
└──────────────────────────────────────────────────────────────────────────"""
class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, 
                 list_IDs, 
                 output_data,
                 aux_input,
                 batch_size=32, 
                 dim=(10,126,201), 
                 n_channels=1,
                 shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.aux_input = aux_input
        self.output_data = output_data
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        #self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indices]

        # Generate data
        SP, w_depth, dzt = self.__data_generation(list_IDs_temp)

        return [SP, w_depth], dzt

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indices = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indices)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        SP = np.empty((self.batch_size, *self.dim, self.n_channels))
        w_depth = np.empty((self.batch_size), dtype=float)
        dzt = np.empty(self.batch_size, dtype=float)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample data
            #print(ID)
            SP[i,] = np.load('NN_data/' + ID + '.npy').reshape(*self.dim,1)
            w_depth[i] = self.aux_input[ID].copy()
            
            # Store class
            dzt[i] = self.output_data[ID].copy()

        return SP, w_depth, dzt
    

"""──────────────────────────────────────────────────────────────────────────┐
│The indexer class generates the indeces to be used as IDs for               │
│DataGenerator_V2. The indexer verifies that the input and output datasets   │
│are numbers (some variables have NaN values in the time series), and the    │
│data is split into training, validation, and test data by year (winter      │
│season from Oct. of that year to Apr. of the following year.)               │
│                                                                            │
└──────────────────────────────────────────────────────────────────────────"""    
def indexer(in_vars = ['SLPN'],
            out_vars = ['ISLH'], 
            in_step = 10,
            out_step = 1,
            split = 0.2,
            test_size = 5,
            data_path = "./NN_data/",
            randomizer = np.random):
    """
    Indexer generates a list of training ids and validation ids to be fed into
    the data generator.
    
    Inputs:
    ----------------------------------------------
    in_vars: list
        List of strings containing the input variables to be considered for 
        indexing; these will be the inputs for the neural network; 4 letter code
        for each variable
    out_vars: list
        List of strings containing the output variables to be considered for 
        indexing; these will be the outputs for the neural network; 4 letter code
        for each variable
    in_step: int
        number of elements to consider for each input var
    out_step: int
        number of elements to consider for each output var
    split: float
        desired split between training and validation data; equal to percentage of
        data to be used for validation.
        *warning* split is approximate, not exact, due to the split being done by
        years.
        
    *warning* The data in the files is assumed to be time continuous. In 
    21AirPolML, this corresponds to a shape[0] = 211 or 212 array representative
    of the days from OCT-01 in the start year given by the year string through
    APRIL-30 the next year. # of days fluctuates due to leap days. This is also
    the reason why the years are stored as separate files and not a multi axis
    np_array
    """
    in_years = dict()
    out_years = dict()
    
    
    if out_step>in_step:
        print("""Error. Output size > Input size. Indexer not configured to 
              process this case""")
        return None
    
    for var in in_vars:
        in_years[var] = []
    
    for var in out_vars:
        out_years[var] = []
    
    #walk through directory to find years associated with data vars
    for root, directories, files in os.walk(data_path):
        for filename in files:
            if filename[-4:] == '.npy':
                var = filename[:4]
                year = filename[5:-4]
                if np.isin(var, in_vars):
                    in_years[var].append(year)
                elif np.isin(var, out_vars):
                    out_years[var].append(year)
    empty_var = False
    for i in in_years.items(): empty_var = empty_var or (len(i[1])==0)
    for i in out_years.items(): empty_var = empty_var or (len(i[1])==0)
    
    #finding intersection of year values
    if not empty_var:
        valid = in_years[list(in_years.keys())[0]]
        for var in list(in_years.keys())[1:]:
            valid = np.intersect1d(valid, in_years[var])
            
        for var in list(out_years.keys()):
            valid = np.intersect1d(valid, out_years[var])
    else:
        print('In_var / Out_var not found. Cancelling')
        return None
    
    #get full list of variables for use in fileloading
    total_vars = []
    total_vars.extend(in_vars)
    total_vars.extend(out_vars)
    
    #load memmap numpy arrays; lazy loading allows use of large files
    
    #initialize a dictionary of valid IDs; key will be years
    valid_ids = dict()
    
    #keeping track of the total number of IDs
    total_size = 0
    
    for year in valid:
        year_ids = []
        loaded_files = dict()
        #check the number of days to be 
        num_days = []
        
        #load the files for the years
        for var in total_vars:
            filename = var + '_'+ str(year) + '.npy'
            filepath = os.path.join(data_path,filename)
            file = np.load(filepath, mmap_mode='r')
            num_days.append(file.shape[0])
            print(filename + ":" + str(file.shape[0]))
            loaded_files[var] = file.copy()
        num_days = np.unique(num_days)
        if num_days.size>1:
            print(f'Inconsistent shape for variables. Data for {year} will',
                  'be skipped')
        elif num_days.size==0 or num_days == 0:
            print(f'Empty variables found. Data for {year} will',
                  'be skipped')
        else:
            #generate list of IDs
            num_itr = int(num_days - in_step)
            
            for i in range(num_itr):
                #bool to track if ID will be dropped
                drop=False
                stop = i + in_step
                out_start = stop - out_step
                
                """
                Generating the ID string. Note that all the information needed
                to load a data sample is contained in the string, except for
                list of data variables. These are not included since these
                will be fed into the data generator using the same parameters
                fed into the indexer
                
                YYYY_DDD_IS_OS
                YYYY: 4 digit year used to look up  file
                DDD: 3 digit day which gives input start_index
                IS: 2 digit step for input data. DDD + IS = stop index for 
                    inputarray
                
                """
                ID_str = f'{year}_{i:03d}_{in_step:02d}_{out_step:02d}'
                
                for in_var in in_vars:
                    array = loaded_files[in_var][i:stop]
                    #print(f'in_array:{array}')
                    drop = (np.any(np.isnan(array)) or drop )
                
                for out_var in out_vars:
                    array = loaded_files[out_var][out_start:stop]
                    #print(f'out_array:{array}')
                    drop = (np.any(np.isnan(array)) or drop )
                
                if ~drop: 
                    #print(ID_str)
                    year_ids.append(ID_str)        
        if len(year_ids)>0:
            valid_ids[year]=year_ids
            total_size += len(year_ids)
    
    #split data into train, validation, and test data. Last year taken as test
    year_list = list(valid_ids.keys())[:-test_size]
    test_years = list(valid_ids.keys())[-test_size:]     
    
    print(f'years:{year_list}')
    
    test = []
    for year in test_years:
        test.extend(valid_ids[year])
        
    #find size of data set minus test set
    work_size = total_size - len(test)
    
    #shuffle the list fo years to iterate through
    randomizer.shuffle(year_list)
    
    train = []
    val = []
    for year in year_list:
        if len(val)/work_size < (split):
            val.extend(valid_ids[year])
        else:
            train.extend(valid_ids[year])
    print(f'Final split: val {len(val)/work_size:.1%} train {len(train)/work_size:.1%}')
    
    
    
    ids = dict()
    ids['train'] = train
    ids['validation'] = val
    ids['test'] = test
    
    return ids


"""──────────────────────────────────────────────────────────────────────────┐
│The DataGenerator_V2 class was built to handle different input variables as │
│channels in the input. It is also built to handle multi-channel output vars,│
│as well as multiple time step output vars. However, multiple time step      │
│output networks have not been tested.                                       │
│The way the data generator is implemented, all input variables must have the│
│same dimensions.                                                             │
└──────────────────────────────────────────────────────────────────────────"""    
class DataGenerator_V2(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, 
                 list_IDs: np.ndarray, 
                 in_vars = ['SLPR'],
                 out_vars = ['TGRA'], 
                 in_step = 10,
                 out_step = 1,
                 batch_size = 32, 
                 shuffle = True,
                 data_path = './NN_data/',
                 randomizer = np.random):
        """
        list_ids expected in YYYY_DDD_IS_OS format. YYYY winter year, DDD day
        along winter, IS input data size, OS output data size 
        """
        
        
        self.data_path = data_path
        
        """
        load a sample of data to extract dimensions for neural network. 
        Implemented only for input data, since we're assuming output variables
        will be flat, not images / multi-d arrays
        """
        sample_id = list_IDs[0]
        sample_path = os.path.join(self.data_path, f'{in_vars[0]}_{sample_id[:4]}.npy')
        sample_in = np.load(sample_path, mmap_mode='r')
        im_size = sample_in.shape[1:]

        self.in_step = in_step
        self.out_step = out_step
        
        dim = (in_step, *im_size)
        
        self.dim = dim
        self.batch_size = batch_size
        self.in_vars = in_vars
        self.out_vars = out_vars
        self.list_IDs = list_IDs
        self.n_channels = len(in_vars)
        self.shuffle = shuffle
        self.randomizer = randomizer
        self.on_epoch_end()


    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indices]

        # Generate data
        in_data, aux_data, out_data = self.__data_generation(list_IDs_temp)

        return [in_data, aux_data], out_data

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indices = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            self.randomizer.shuffle(self.indices)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        in_data = np.empty((self.batch_size, *self.dim, self.n_channels))
        aux_data = np.empty((self.batch_size), dtype=float)
        out_data = np.empty((self.batch_size, len(self.out_vars)*self.out_step), dtype=float)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            """
            aux_data is a measure of how far into the winter time series the
            measurement is. Ideally ranged from 0 to 1 using 
                sin(pi * day_index /days_in_winter )
            where days_in_winter varies from 200 to 201 (number of days from
            october 10th through april 30th) due to leap years.
            However, for simplicity this is calculated considering 
            days_in_winter = 201. 
            """
            day_index = int(ID[5:8])
            aux_data[i] = np.sin(np.pi * day_index / (212-self.in_step))
            
            # Store sample data
            #print(ID)
            path = os.path.join(self.data_path, f'{self.in_vars[0]}_{ID[:4]}.npy')
            
            #reshape in order to be able to stack easily with numpy
            temp_in = np.load(path, mmap_mode='r')[day_index:day_index+self.in_step].reshape(1,*self.dim)

            if len(self.in_vars)>1:
                for var in self.in_vars[1:]:
                    path = os.path.join(self.data_path, f'{var}_{ID[:4]}.npy')
                    temp_data = np.load(path, mmap_mode='r')[day_index:day_index+self.in_step].reshape(1,*self.dim)
                    temp_in = np.vstack((temp_in,temp_data))
            in_data[i,] = np.moveaxis(temp_in, 0, -1).copy()
            
            # output data
            path = os.path.join(self.data_path, f'{self.out_vars[0]}_{ID[:4]}.npy')
            out_idx = day_index + self.in_step - self.out_step
            temp_out = np.load(path)[out_idx:out_idx+self.out_step].reshape(1,self.out_step)
            if len(self.out_vars)>1:
                for var in self.out_vars[1:]:
                    path = os.path.join(self.data_path, f'{var}_{ID[:4]}.npy')
                    temp_data = np.load(path, mmap_mode='r')[out_idx:out_idx+self.out_step].reshape(1,self.out_step)
                    temp_out = np.vstack((temp_out,temp_data))
            out_data[i,] = np.moveaxis(temp_out, 0, -1).flatten().copy()
                    
        #gc.collect()
        return in_data, aux_data, out_data
