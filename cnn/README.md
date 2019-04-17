# Convolutional Neural Network


To train the CNN from scratch, we first need to create the selective indexing dataset and store it in a MySQL database.


## Prerequisites:


- Ubuntu 16.04
- MySQL Community Server version 5.7.25
- Anaconda 5.2.0
- NVIDIA GeForce GTX 1080 Ti (11GB GPU memory) (Driver version 410.79)


## 1) Creating and activating an anaconda environment

```
conda create --name selective_indexing --file requirements.txt
source activate selective_indexing
```

## 2) Creating the required folder structure

<!-- language: lang-none -->
    \root-dir
        - downloaded-data
        - extracted-data
        \preprocessed
            - selective-indexing
            - cross-validation
            - vocab

## 3) Extracting journal indexing periods

The scripts for steps 3-6 are in the cnn/scripts folder.

Copy _aux/lsi2018.xml to root-dir/lsi2018.xml

```
python extract_journal_indexing_periods.py
```

The output file of interest is root-dir/preprocessed/selective-indexing/selective_indexing_periods_output.txt. The file contains values delimited by |. All the data is extracted from the lsi2018.xml file. The start and end years of selective indexing are extracted from the Coverage element in the indexing history.

The _aux/selective_indexing_periods.csv file contains data for the journal_indexing_periods table in the database. It is created from the selective_indexing_periods_output.txt file, which we just generated. The CSV file has four columns: the journal NLM id, the citation subset, the start year of selective indexing, and the end year of selective indexing (-1 if currently indexed). Some of the start/end years of selective indexing are modified based on the text in the CoverageNote element of the lsi2018.xml file.

## 4) Creating the database

First, enter the root directory of the created folder structure in the cnn/scripts/settings.py file (directory should not end with a forward slash).
    
### a) Create the database schema

run the create_empty_database.sql script.

### b) Prepopulate database tables

Configure the database host, user and password in the cnn/scripts/settings.py  file.

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

### c) Load article data

i. Download the MEDLINE baseline 

```
python download_medline_data.py
```

ii. Copy Novemeber 2017 - September 2018 daily update files (#929-#1250) to root-dir/downloaded-data. These files can be downloaded with the GitHub releases.

iii. Extract required data from baseline files

```
python extract_data.py
```

iv. Load article data into the database

```
python load_data.py
```

## 5) Create train, validation, and test sets


### a) Create the datasets

```
python create_crossvalidation_sets.py
```

The train, validation, and test sets are created in the root-dir/preprocessed/cross-validation directory. Note that we call the validation sets: "dev sets". The "dev set" contains articles from all years, while the "target dev set" only contains articles from 2018.

Due to the size of the database; this script may take a long time to run.

### b) Clean train and dev sets

As mentioned in the paper, some journals are known to have problematic determinations before 2015. To remove this data from the train and dev sets:

Copy the _aux/problematic_journals.csv file to root-dir/preprocessed/selective-indexing/problematic_journals.csv

```
python clean_dev_and_train_sets.py
```

### c) Create final reporting test set

This test set only contains articles from selectively indexed journals of interest to NLM indexers.

Copy the _aux/selectively-indexed-journals-of-interest.csv file to root-dir/preprocessed/selective-indexing/selectively-indexed-journals-of-interest.csv

```
python create_reporting_test_set.py
```

## 6) Create the vocabulary

This step creates a dictionary of word to index mappings.

You may first need to download the supporting data for the NLTK tokenizer. In a Python REPL run:

```
import nltk
nltk.download('punkt')
```

Now, create the vocabulary:

```
python create_word_index_lookup.py
```

## 7) Training the model

The scripts to train and test the CNN model are in the ./cnn folder.

In the machine_settings.py file set the data_dir, database_host, and runs_dir values. The data_dir should be the root directory that was created earlier.

In the cnn/setting.py file add the connection configuration for the previously created MySQL database.

```
python train.py
```

The training output will be saved in the previously specified runs_dir (machine_settings.py) and the folder name will be a numerical timestamp.

## 8) Make pipeline validation and test set predictions

i. In the cnn/settings.py file update the sub_dir field of the restore config to the name of the folder containing the training output (e.g. 1553531627). Also update the threshold field to the optimum threshold determined during training (see output-dir/logs/best_epoch_logs.txt - val_threshold).

ii. Next, set the pmids_filepath field of the pred config to the full/relative path of the datasets/pipeline_validation_set.json file. Also change the results_dir field to predictions_val. Then run:

```
python pred.py
```

iii. Finally, set the pmids_filepath field of the pred config to the full/relative path of the datasets/pipeline_test_set.json file. Also change the results_dir field to predictions_test. Then run:

```
python pred.py
```

Two additional folders (predictions_val and predictions_test) will now have been created in the training output folder. These folders contain dereferenced_predictions.csv files that contain the model predictions for the pipeline validation and test sets. 