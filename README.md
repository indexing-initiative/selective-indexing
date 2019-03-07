# A High Recall Classifier for Selecting Articles for MEDLINE Indexing

This repository contains the code, datasets, and trained models required to reproduce the results of the paper "A High Recall Classifier for Selecting Articles for MEDLINE Indexing" by Alastair Rae, Max Savery, James Mork, and Dina Demner-Fushman. 

The list of selectively indexed NLM journal IDs that are of interest to NLM indexer can be downloaded from ./_aux/selectively-indexed-journals-of-interest.csv

To reproduce the paper results from scratch you need to:

1. Train the voting ensemble model. See the README file in the voting folder for instructions.
2. Train the CNN model. See the README file in the cnn folder for instructions.
3. Generate predictions for both models for the validation and test sets. The validation and test set data is available in the datasets folder.
4. Follow the instructions in the README file in the results folder to generate the paper results.