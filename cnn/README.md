# Convolutional Neural Network

To train the CNN from scratch, we first need to create the selective indexing dataset and store it in a MySQL database.

## Prerequisites:

- Ubuntu 16.04
- MySQL Community Server version 5.7.25
- Anaconda 5.2.0
- NVIDIA GeForce GTX 1080 Ti (11GB GPU memory)(Driver version 410.79)

## Creating and activating an anaconda environment
```
conda create --name selective_indexing --file requirements.txt
source activate selective_indexing
```
## Creating the required folder structure

<!-- language: lang-none -->
    \root-dir
        - downloaded-data
    
## Creating the MySQL database

The scripts to create the MySQL database are in the cnn/scripts folder

1. Enter the root directory of the created folder structure in the settings.py file (directory should not end with a forward slash)
2. Download the MEDLINE baseline
```
python download_medline_data.py
```


