┌───────────────────────────────────────────────────────────────────────────────┐
│This is the NN input and target data folder for the 21AIRPOLML Project.	│
│-----------									│
│Filenames include a 4 letter identifier and the winter season associated with	│
│them (e.g., 2000 includes data from October 1st, 2000 through April 30th, 2001.│ 
│										│
│Data has been stored as a numpy array, so use np.load(filename) to load into	│
│python.									│
│										│
│Each variable has been normalized, generally to values between 0 and 1. In	│
│order to obtain the original values, the <varname>_reconstructor.pkl, which	│
│contains a dictionary including the minimum value (key:'min') and the range of │
│values (key: 'range'). 							│
│In order to obtain the original values, load the data with numpy and the 	│  
│reconstructor with pickle, and calculate as:					│
│original_val = norm_val * reconstructor['range'] + reconstructor['min']	│
│										│
│List of available data:							│
│										│
│ISLH - In situ low-high temperature gradient. Calculated using Col de Porte	│
│station data and Pont de Claix temperature data.				│
│										│
│ISLM - In situ low-mid temperature gradient. Calculated using Peuil de Claix	│
│and Pont de Claix temperature data.						│
│										│
│ISDC - In situ daily change in vertical temperature gradient. Calculated from	│
│ISLH.										│
│										│
│SLPN - Sea Level Pressure, Normalized. These are the MAR forced by ERA5 Sea	│
│level pressure fields. For reconstructor, see SSEN.				│
│										│
│SSEN - Sea Level Pressure, Normalized using ERA5 range values. These are the 	│
│sea level pressure fields from MAR forced by MPI-ESM1.2, normalized using the	│
│data availale for MAR forced by ERA5 (see SLPN). SSEN reconstructor can be 	│
│applied to SLPN to retrieve values.						│
│										│
└───────────────────────────────────────────────────────────────────────────────┘

