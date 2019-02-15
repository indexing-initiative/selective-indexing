from contextlib import closing
from mysql.connector import connect
import random


def create_journal_group_name_lookup(filepath, encoding, delimiter):
    data = load_delimited_data(filepath, encoding, delimiter)
    lookup = {}
    for row in data:
        nlm_id = row[0]
        group = row[1]
        lookup[nlm_id] = group
    return lookup


def create_id_lookup(db_config, sql):
    lookup = {}
    with closing(connect(**db_config)) as conn:
        with closing(conn.cursor()) as cursor:    #pylint: disable=E1101
            cursor.execute(sql)                   #pylint: disable=E1101
            for row in cursor.fetchall():         #pylint: disable=E1101
                id, ui = row
                lookup[ui] = id
    return lookup


def load_delimited_data(path, encoding, delimiter):
    with open(path, 'rt', encoding=encoding) as file:
        data = tuple( tuple(data_item.strip() for data_item in line.strip().split(delimiter)) for line in file ) 
    return data


def load_ids_from_file(path, encoding):
    ids = [int(id[0]) for id in load_delimited_data(path, encoding, ',')]
    return ids


def load_indexing_periods(filepath, encoding, is_fully_indexed):
    periods = {}
    with open(filepath, 'rt', encoding=encoding) as file:
        for line in file:
            split = line.split(',')

            nlm_id = split[0].strip()
            citation_subset = split[1].strip()
            start_year = int(split[2].strip())
            end_year = int(split[3].strip())
            
            if start_year < 0:
                continue
            if end_year < 0:
                end_year = None

            period = { 'citation_subset': citation_subset, 'is_fully_indexed': is_fully_indexed, 'start_year': start_year, 'end_year': end_year }
            if nlm_id in periods:
                periods[nlm_id].append(period)
            else:
                periods[nlm_id] = [period]
    return periods


def random_permutation(iterable, r=None):
    pool = tuple(iterable)
    r = len(pool) if r is None else r
    return tuple(random.sample(pool, r))


def save_delimited_data(path, encoding, delimiter, data):
    with open(path, 'wt', encoding=encoding) as file:
        for data_row in data:
            line = delimiter.join([str(data_item) for data_item in data_row]) + '\n'
            file.write(line)


def should_review_coverage_note(coverage_note_text):
    coverage_note_text_lower = coverage_note_text.lower()
    should_review = str('sel' in coverage_note_text_lower or 'ful' in coverage_note_text_lower)
    return should_review


def write_ids_to_file(path, encoding, ids):
    save_delimited_data(path, encoding, ',', [(id,) for id in ids])