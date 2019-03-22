from contextlib import closing
from helper import load_delimited_data, load_ids_from_file, random_permutation, write_ids_to_file
from mysql.connector import connect
from settings import get_config

SELECT_ARTICLE_DATA_SQL = 'SELECT j.nlm_id, a.pub_year FROM articles AS a, journals AS j WHERE a.journal_id = j.id AND a.id = %(id)s'
YEAR_PROBLEM_FIXED = 2014

def run(config):
    ENCODING = config['encoding']
    DB_CONFIG = config['database']
    PROBLEMATIC_JOURNALS_FILEPATH = config['problematic_journals_file']
    
    DEV_SET_FILEPATH = config['dev_set_ids_file']
    TRAIN_SET_FILEPATH = config['train_set_ids_file']

    CLEANED_DEV_SET_FILEPATH = config['cleaned_dev_set_ids_file']
    CLEANED_TRAIN_SET_FILEPATH = config['cleaned_train_set_ids_file']
  
    problematic_nlm_ids = [nlm_id for nlm_id, _ in load_delimited_data(PROBLEMATIC_JOURNALS_FILEPATH, ENCODING, ',')]
    
    dev_set_ids = load_ids_from_file(DEV_SET_FILEPATH, ENCODING)
    cleaned_dev_set_ids = _clean_dataset(dev_set_ids, problematic_nlm_ids, DB_CONFIG)
    cleaned_dev_set_ids = random_permutation(cleaned_dev_set_ids)
    write_ids_to_file(CLEANED_DEV_SET_FILEPATH, ENCODING, cleaned_dev_set_ids)

    train_set_ids = load_ids_from_file(TRAIN_SET_FILEPATH, ENCODING)
    cleaned_train_set_ids = _clean_dataset(train_set_ids, problematic_nlm_ids, DB_CONFIG)
    cleaned_train_set_ids = random_permutation(cleaned_train_set_ids)
    write_ids_to_file(CLEANED_TRAIN_SET_FILEPATH, ENCODING, cleaned_train_set_ids)


def _clean_dataset(dataset_ids, problematic_nlm_ids, db_config):
    cleaned_ids = []
    with closing(connect(**db_config)) as conn, closing(conn.cursor()) as cursor:    #pylint: disable=E1101
        for id in dataset_ids:
            cursor.execute(SELECT_ARTICLE_DATA_SQL, {'id': id})    #pylint: disable=E1101
            nlm_id, pub_year = cursor.fetchone()    #pylint: disable=E1101
            if nlm_id in problematic_nlm_ids and pub_year <= YEAR_PROBLEM_FIXED:
                continue
            cleaned_ids.append(id)
    return cleaned_ids

if __name__ == '__main__':
    config = get_config()
    run(config)