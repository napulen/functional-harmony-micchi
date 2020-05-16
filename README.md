# Not All Roads Lead to Rome: Pitch Representation and Model Architecture for Automatic Harmonic Analysis
This repository contains the code that has been used to train and evaluate the models described in the TISMIR paper https://transactions.ismir.net/articles/10.5334/tismir.45/ .
We invite you to consult the paper (which is open access) for more information on the models and data representations used.

Please contact us at gianluca.micchi@univ-lille.fr for further information.

## Code explanation
There are several entry points:
 - run\_full.py, that can be used to run a previously trained model with symbolic music files as input; a pre-trained model (run\_model.h5) is alreadt provided.
 - train.py, that can be used to train new models, provided they are in encoded in the correct format
 - converters.py, which allows for conversion between the three different supported file formats: tabular, rntxt, and dezrann
 - data_manipulation/preprocessing.py, that can be used to encode data from tabular format into tfrecords for training
 - analyse_results.py, that can be used to get some plots and insight into the results of a model by comparing the predictions with the targets
