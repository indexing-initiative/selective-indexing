from contextlib import closing
from datetime import datetime as dt
import gzip
from helper import create_id_lookup
import json
from mysql.connector import connect
from mysql.connector.errors import IntegrityError
from settings import get_config


SELECT_ALL_JOURNALS_SQL = 'SELECT id, nlmid FROM journals'
INSERT_ARTICLE_SQL = 'INSERT INTO articles (id, pmid, title, abstract, pub_year, date_completed, journal_id, is_indexed) VALUES (%(id)s, %(pmid)s, %(title)s, %(abstract)s, %(pub_year)s, %(date_completed)s, %(journal_id)s, %(is_indexed)s)'
SELECT_ARTICLE_ID_BY_PMID_SQL = 'SELECT id from articles WHERE pmid = %(pmid)s'
DELETE_ARTICLE_BY_ID_SQL = 'DELETE FROM articles WHERE id = %(id)s'
SELECT_ALL_REF_TYPES = 'SELECT id, name FROM ref_types'
INSERT_ARTICLE_REF_TYPE_SQL = 'INSERT INTO article_ref_types (article_id, ref_type_id) VALUES (%(article_id)s, %(ref_type_id)s)'
DELETE_REF_TYPES_BY_ARTICLE_ID_SQL = 'DELETE FROM article_ref_types WHERE article_id = %(article_id)s'


def run(config):
    ENCODING = config['encoding']
    JSON_FILEPATH_TEMPLATE = config['json_filepath_template']
    START_DATA_FILE_NUM = config['start_data_file_num']
    END_DATA_FILE_NUM = config['end_data_file_num']
    START_ID = 1
    DB_CONFIG = config['database']
    MAX_ABS_LEN = config['max_abs_len']
    LOG_FILEPATH = config['load_data_log_file']

    with open(LOG_FILEPATH, 'wt', encoding=ENCODING) as log_file, \
        closing(connect(**DB_CONFIG)) as conn:

        journal_id_lookup = create_id_lookup(DB_CONFIG, SELECT_ALL_JOURNALS_SQL)
        ref_type_id_lookup = create_id_lookup(DB_CONFIG, SELECT_ALL_REF_TYPES)
        article_id = START_ID
        for file_num in range(START_DATA_FILE_NUM, END_DATA_FILE_NUM + 1):
            print(file_num)
            json_filepath = JSON_FILEPATH_TEMPLATE.format(file_num)
            with gzip.open(json_filepath, 'rt', encoding=ENCODING) as json_file: 
                article_data = json.load(json_file)
                for article in article_data['articles']:
                    next_id = _insert_new_article(conn, journal_id_lookup, ref_type_id_lookup, article_id, article, MAX_ABS_LEN, log_file)
                    article_id = next_id
            conn.commit()   #pylint: disable=E1101


def _insert_article_row(conn, data):
    with closing(conn.cursor()) as cursor:
        cursor.execute(INSERT_ARTICLE_SQL, data)    #pylint: disable=E1101


def _insert_ref_types(conn, data):
    with closing(conn.cursor()) as cursor:
        for article_id, ref_type_id in data:
            cursor.execute(INSERT_ARTICLE_REF_TYPE_SQL, { 'article_id': article_id, 'ref_type_id': ref_type_id})    #pylint: disable=E1101


def _insert_new_article(conn, journal_id_lookup, ref_type_id_lookup, id, article, max_abs_len, log_file):
    # Prepare data
    pmid = article['pmid']
    title = article['title']
    abstract = article['abstract']
    abstract_len = len(abstract)
    if abstract_len > max_abs_len:
        abstract = abstract[:max_abs_len]
        log_file.write('{}: abstract length {} truncated to {}\n'.format(pmid, abstract_len, max_abs_len))
        log_file.flush()
    pub_year = article['pub_year']
    date_completed = dt.strptime(article['date_completed'], '%Y-%m-%d').date() # Assume ISO date format
    journal_nlmid = article['journal_nlmid']
    journal_id = journal_id_lookup[journal_nlmid] if (journal_nlmid in journal_id_lookup) else None
    is_indexed = article['is_indexed']
    ref_types = article['ref_types']
    article_row_data = { 'id': id, 'pmid':  pmid, 'title': title, 'abstract': abstract, 'pub_year': pub_year, 'date_completed': date_completed, 'journal_id': journal_id, 'is_indexed': is_indexed }
    ref_type_data = [(id, ref_type_id_lookup[ref_type]) for ref_type in ref_types if ref_type in ref_type_id_lookup]
    try:    
        _insert_article_row(conn, article_row_data)
        _insert_ref_types(conn, ref_type_data)
        next_id = id + 1
    except IntegrityError as ex:
        if ex.errno == 1062 and 'pmid_UNIQUE' in ex.msg:
            _replace_existing_article(conn, pmid, article_row_data, ref_type_data, log_file)
            next_id = id
    return next_id


def _replace_existing_article(conn, pmid, article_row_data, ref_type_data, log_file):
    with closing(conn.cursor()) as cursor:
         cursor.execute(SELECT_ARTICLE_ID_BY_PMID_SQL, {'pmid': pmid})    #pylint: disable=E1101
         existing_id = cursor.fetchone()[0]                           #pylint: disable=E1101

    with closing(conn.cursor()) as cursor:
         cursor.execute(DELETE_REF_TYPES_BY_ARTICLE_ID_SQL, {'article_id': existing_id})    #pylint: disable=E1101

    with closing(conn.cursor()) as cursor:
         cursor.execute(DELETE_ARTICLE_BY_ID_SQL, { 'id': existing_id })    #pylint: disable=E1101
    
    article_row_data['id'] = existing_id
    ref_type_data = [(existing_id, ref_type_id) for _, ref_type_id, in ref_type_data]
    _insert_article_row(conn, article_row_data)
    _insert_ref_types(conn, ref_type_data)

    log_file.write('Duplicate pmid {}: {} id replaced\n'.format(pmid, existing_id))
    log_file.flush()


if __name__ == '__main__':
    config = get_config()
    run(config)
