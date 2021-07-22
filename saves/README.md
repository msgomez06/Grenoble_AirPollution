╭────────────────────────────────────────────────────────────────────────────╮
│Models trained with the network are stored in this location by default.     │
│                                                                            │
│params_{time_string}.pkl stores the global parameters dictionary associated │
│with the model, and model_{time_str}.h5 stores the keras model.             │
│                                                                            │
│params loaded with regular pickle handler. Model loaded with                │
│keras.models.load_model(f'{path_to_model}')                                 │
└────────────────────────────────────────────────────────────────────────────╯
