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
            - cross-validation
            - vocab

## Extracting journal indexing periods

Copy _aux/lsi2018.xml to root-dir/lsi2018.xml

```
python extract_journal_indexing_periods.py
```

The output file of interest is root-dir/preprocessed/selective-indexing/selective_indexing_periods_output.txt. The file contains values delimited by |. All the data is extracted from the lsi2018.xml file. The start and end years of selective indexing are extracted from the Coverage element in the indexing history.

The _aux/selective_indexing_periods.csv file contains data for the journal_indexing_periods table in the database. It is created from the selective_indexing_periods_output.txt file, which we just generated. The CSV file has four columns: the journal NLM id, the citation subset, the start year of selective indexing, and the end year of selective indexing (-1 if currently indexed). Some of the start/end years of selective indexing are modified based on the text in the CoverageNote element of the lsi2018.xml file.

## Creating the database

The scripts to create and load data into the MySQL database are in the cnn/scripts folder

First, enter the root directory of the created folder structure in the settings.py file (directory should not end with a forward slash)
    
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
Copy _aux/selective_indexing_periods_input.csv to root-dir/preprocessed/selective-indexing/selective_indexing_periods_input.csv
```
python populate_journal_indexing_periods_table.py
```
## Load article data

1. Download the MEDLINE baseline 
```
python download_medline_data.py
```
2. Copy January - September 2018 daily update files (TODO: how to share daily update files) to root-dir/downloaded-data
3. Extract required data from baseline files
```
python extract_data.py
```
4. Load article data into the database
```
python load_data.py
```

## Create train, validation, and test sets

```
python create_crossvalidation_sets.py
```

The train, validation, and test sets are created in the root-dir/preprocessed/cross-validation directory. Note that we call the validation sets: "dev sets". The "dev set" contains articles from all years, while the "target dev set" only contains articles from 2018.

Due to the size of the database; this script may take a long time to run.

## Clean train and dev sets

As mentioned in the paper, some journals are known to have problematic determinations before 2015. To remove this data from the train and dev sets:

1. Copy the _aux/problematic_journals.csv file to root-dir/preprocessed/selective-indexing/problematic_journals.csv

2. Run the following Python script:
```
python clean_dev_and_train_sets.py
```

## Create final reporting test set

This test set only contains articles from selectively indexed journals of interest.

1. Copy the _aux/selectively-indexed-journals-of-interest.csv file to root-dir/preprocessed/selective-indexing/selectively-indexed-journals-of-interest.csv

2. Run:
```
python create_reporting_test_set.py
```

## Create the vocabulary

This step creates a dictionary of word-index mappings.

1. You may first need to download the supporting data for the NLTK tokenizer. In a Python REPL run:
```
import nltk
nltk.download('punkt')
```
2. Next run:
```
python create_word_index_lookup.py
```