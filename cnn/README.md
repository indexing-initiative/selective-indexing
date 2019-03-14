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
        - extracted-data
        \preprocessed
            - selective-indexing

## Extracting journal indexing periods

Copy _aux/lsi2018.xml to root-dir/lsi2018.xml

```
python extract_journal_indexing_periods.py
```

The output file of interest is root-dir/preprocessed/selective-indexing/selective_indexing_periods_output.txt. The file contains values delimited by |. All the data is extracted from the lsi2018.xml file. The start and end years of selective indexing are extracted from the Coverage element in the indexing history.

The _aux/selective_indexing_periods.csv file contains data for the journal_indexing_periods table in the database. It is created from the selective_indexing_periods_output.txt file, which we just generated. The CSV file has four columns: the journal NLM id, the citation subset, the start year of selective indexing, and the end year of selective indexing (-1 if currently indexed). Some of the start/end years of selective indexing are modified based on the text in the CoverageNote element of the lsi2018.xml file.

## Creating the database

The scripts to create and load data into the MySQL database are in the cnn/scripts folder
    
## Creating the database schema

run the create_empty_database.sql script

## Prepopulate database tables

Configure database host, user and password in the settings file.

```
python populate_ref_types_table.py
```
```
python populate_journal_groups_table.py
```

Copy _aux/J_Medline.txt to root-dir/J_Medline.txt

Copy _aux/selectively_indexed_journal_groups.csv to root-dir/preprocessed/selective-indexing/selectively_indexed_journal_groups.csv
```
python populate_journals_table.py
```

## Load citation data

The scripts to create and load data into the MySQL database are in the cnn/scripts folder

1. Enter the root directory of the created folder structure in the settings.py file (directory should not end with a forward slash)
2. Download the MEDLINE baseline (TODO: how to share daily update files)
```
python download_medline_data.py
```

3. Extract required data from baseline files
```
python extract_data.py
```



