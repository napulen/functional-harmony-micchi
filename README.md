# Not All Roads Lead to Rome: Pitch Representation and Model Architecture for Automatic Harmonic Analysis
This repository contains the code that has been used to train and evaluate the models described in the TISMIR paper https://transactions.ismir.net/articles/10.5334/tismir.45/ .
We invite you to consult the paper (which is open access) for more information on the models and data representations used.

The corpus in this repository reflects the one used at the moment of the paper submission. You can find the most updated version at https://github.com/MarkGotham/When-in-Rome.

UPDATE: The repository contains the updated scripts and model related to the ISMIR2021 paper: [A deep learning method for enforcing coherence in Automatic Chord Recognition](https://archives.ismir.net/ismir2021/paper/000055.pdf).

Please contact me at gianluca.micchi@gmail.com for further information.

## (brief) Code explanation
The main entry points to the code are stored under the folder scripts. Briefly:
 - cra_preprocess: encode data from tabular format into tfrecords for training; `python3 -m scripts.cra_preprocess /data` creates the tfrecords from the data files contained in the folder.
 - cra_train: train a new model; `python3 -m scripts.cra_train <ARGUMENTS>`, launch with flag -h to get help 
 - cra_run: run the model on a symbolic music score file and return its harmonic analysis
 - cra_optimisation: run a search for the best hyperparameters