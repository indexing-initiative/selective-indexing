from contextlib import closing
from .helper import create_journal_group_name_lookup, load_delimited_data, save_delimited_data
from mysql.connector import connect
from .settings import get_config


SELECT_JOURNAL_DATA_BY_ID = 'SELECT nlmid, medline_ta from journals WHERE id = %(id)s'
SELECT_ARTICLE_DATA_BY_ID = 'SELECT pmid, pub_year from articles WHERE id = %(id)s'


def run(predictions_filepath, journal_groups_filepath, db_config, dereferenced_predictions_filepath, encoding, delimiter):
    predictions = load_delimited_data(predictions_filepath, encoding, delimiter)
    with closing(connect(**db_config)) as db_conn:
        journal_group_lookup = create_journal_group_name_lookup(journal_groups_filepath, encoding, delimiter)
        dereferenced = _dereference_predictions(db_conn, predictions, journal_group_lookup)
    save_delimited_data(dereferenced_predictions_filepath, encoding, delimiter, dereferenced)


def _dereference_predictions(db_config, predictions, journal_group_lookup):
    dereferenced = []
    for prediction in predictions:
        article_id, journal_id, act, score = prediction
        pmid, pub_year = _get_by_id(db_config, SELECT_ARTICLE_DATA_BY_ID, article_id)
        journal_info = _get_by_id(db_config, SELECT_JOURNAL_DATA_BY_ID, journal_id)
        nlmid, medline_ta = 'unknown', 'unknown'
        if journal_info:
             nlmid, medline_ta = journal_info
        journal_group = journal_group_lookup[nlmid] if nlmid in journal_group_lookup else 'unknown'
        dereferenced.append((pmid, pub_year, nlmid, medline_ta, journal_group, act, score))
    return tuple(dereferenced)


def _get_by_id(db_conn, query, id):
    with closing(db_conn.cursor()) as cursor:    #pylint: disable=E1101
        cursor.execute(query, { 'id': int(id) })    #pylint: disable=E1101
        return cursor.fetchone()    #pylint: disable=E1101


if __name__ == '__main__':
    predictions_filepath = '/****/predictions.csv'
    dereferenced_predictions_filepath = '/****/dereferenced_predictions.csv'
    encoding = 'utf8'
    delimiter = ','
    config = get_config()
    db_config = config['database']
    journal_groups_filepath = config['journal_groups_file']
    run(predictions_filepath, journal_groups_filepath, db_config, dereferenced_predictions_filepath, encoding, delimiter)
