import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contextlib import closing
from data_helper import load_db_ids, create_word_index_lookup
from mysql.connector import connect
from pickle import dump
from settings import get_config


def run(config):
    ENCODING = config['encoding']
    TRAIN_SET_IDS_FILEPATH = config['cleaned_train_set_ids_file']
    DB_CONFIG = config['database']
    WORD_INDEX_LOOKUP_FILEPATH = config['word_index_lookup_file']

    db_ids = load_db_ids(TRAIN_SET_IDS_FILEPATH, ENCODING)

    with closing(connect(**DB_CONFIG)) as db_conn:
        word_index_lookup = create_word_index_lookup(db_conn, db_ids, verbose=True)

    with open(WORD_INDEX_LOOKUP_FILEPATH, 'wb') as save_file:
        dump(word_index_lookup, save_file)


if __name__ == '__main__':
    config = get_config()
    run(config)