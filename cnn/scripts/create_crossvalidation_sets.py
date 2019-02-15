from contextlib import closing
from helper import load_ids_from_file, random_permutation, write_ids_to_file
from mysql.connector import connect
from settings import get_config


SELECT_ARTICLE_IDS = 'SELECT DISTINCT a.id FROM articles AS a INNER JOIN journal_indexing_periods AS p ON a.journal_id = p.journal_id LEFT JOIN article_ref_types AS art ON a.id = art.article_id WHERE NOT p.is_fully_indexed AND a.pub_year > p.start_year AND (a.pub_year < p.end_year OR p.end_year IS NULL) AND a.pub_year >= %(start_year)s AND a.pub_year <= %(end_year)s AND (art.ref_type_id NOT IN (3, 6, 8, 9, 11, 14, 16, 17) OR art.ref_type_id IS NULL)'


def run(config):
    USE_EXISTING = config['cross_val_use_existing']
    DB_CONFIG = config['database']
    ENCODING = config['encoding']
    excluded_ids = []

    # Test
    START_YEAR = config['test_set_start_year'] 
    END_YEAR = config['test_set_end_year'] 
    SIZE = config['test_set_size'] 
    SAVE_FILEPATH = config['test_set_ids_file']
    if USE_EXISTING:
        excluded_ids = load_ids_from_file(SAVE_FILEPATH, ENCODING)
    else:
        excluded_ids = _create_set(DB_CONFIG, START_YEAR, END_YEAR, SIZE, SAVE_FILEPATH, ENCODING, excluded_ids)

    # Target Dev
    START_YEAR = config['target_dev_set_start_year'] 
    END_YEAR = config['target_dev_set_end_year'] 
    SIZE = config['target_dev_set_size'] 
    SAVE_FILEPATH = config['target_dev_set_ids_file']
    excluded_ids = _create_set(DB_CONFIG, START_YEAR, END_YEAR, SIZE, SAVE_FILEPATH, ENCODING, excluded_ids)
    
    # Dev
    START_YEAR = config['dev_set_start_year'] 
    END_YEAR = config['dev_set_end_year'] 
    SIZE = config['dev_set_size'] 
    SAVE_FILEPATH = config['dev_set_ids_file']
    excluded_ids = _create_set(DB_CONFIG, START_YEAR, END_YEAR, SIZE, SAVE_FILEPATH, ENCODING, excluded_ids)

    # Train
    START_YEAR = config['train_set_start_year'] 
    END_YEAR = config['train_set_end_year'] 
    SIZE = config['train_set_size'] 
    SAVE_FILEPATH = config['train_set_ids_file']
    _create_set(DB_CONFIG, START_YEAR, END_YEAR, SIZE, SAVE_FILEPATH, ENCODING, excluded_ids)

    
def _create_set(db_config, start_year, end_year, size, save_filepath, save_encoding, excluded_ids):
    period_ids = _get_ids(db_config, start_year, end_year)
    set_ids = set(period_ids) - set(excluded_ids)
    set_ids = random_permutation(set_ids)
    set_ids = set_ids[:size]
    
    write_ids_to_file(save_filepath, save_encoding, set_ids)

    updated_excluded_ids = list(excluded_ids)
    updated_excluded_ids.extend(set_ids)
    return updated_excluded_ids

 
def _get_ids(db_config, start_year, end_year):
    ids = []
    with closing(connect(**db_config)) as db_conn:
        with closing(db_conn.cursor()) as cursor:                                   #pylint: disable=E1101
            cursor.execute(SELECT_ARTICLE_IDS, { 'start_year' : start_year, 'end_year' : end_year })    #pylint: disable=E1101
            for row in cursor.fetchall():                                           #pylint: disable=E1101
                id = row[0]
                ids.append(id)
    return ids


if __name__ == '__main__':
    config = get_config()
    run(config)