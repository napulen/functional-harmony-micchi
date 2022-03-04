# Not All Roads Lead to Rome: Pitch Representation and Model Architecture for Automatic Harmonic Analysis
This repository contains the code that has been used to train and evaluate the models described in the TISMIR paper https://transactions.ismir.net/articles/10.5334/tismir.45/ .
We invite you to consult the paper (which is open access) for more information on the models and data representations used.

The corpus in this repository reflects the one used at the moment of the paper submission. You can find the most updated version at https://github.com/MarkGotham/When-in-Rome.

UPDATE: The repository contains the updated scripts and model related to the ISMIR2021 paper: [A deep learning method for enforcing coherence in Automatic Chord Recognition](https://archives.ismir.net/ismir2021/paper/000055.pdf).


Please contact me at gianluca.micchi@gmail.com for further information.

## Code explanation
There are several entry points:
 - run\_full.py, that can be used to run a previously trained model with symbolic music files as input; a pre-trained model (run\_model.h5) is already provided; `python3 run_full.py --in /PATH/TO/SCORE` runs the model and analyses the given score
 - train.py, that can be used to train new models, provided they are in encoded in the correct format; `python3 train.py --model 0 --input 4` trains a model with the same parameters as in the paper
 - converters.py, which allows for conversion between the three different supported file formats: tabular, rntxt, and dezrann
 - data_manipulation/preprocessing.py, that can be used to encode data from tabular format into tfrecords for training; `python3 preprocessing.py /PATH/TO/FOLDER` creates the tfrecords from the file contained in the folder. The folder should have a specific structure: two subfolders, train and valid, each with two subfolders, chords and scores.
 - analyse_results.py, that can be used to get some plots and insight into the results of a model by comparing the predictions with the targets
