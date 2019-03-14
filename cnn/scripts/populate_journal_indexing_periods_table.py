from contextlib import closing
from helper import load_indexing_periods, create_id_lookup
from mysql.connector import connect
from mysql.connector.errors import IntegrityError
from settings import get_config


SELECT_ALL_JOURNALS_SQL = 'SELECT id, nlmid FROM journals'
INSERT_SQL = 'INSERT INTO journal_indexing_periods (journal_id, citation_subset, is_fully_indexed, start_year, end_year) VALUES (%(journal_id)s, %(citation_subset)s, %(is_fully_indexed)s, %(start_year)s, %(end_year)s)'


def run(config):
    SELECTIVE_INDEXING_PERIODS_FILEPATH = config['selective_indexing_periods_input_file']
    #FULL_INDEXING_PERIODS_FILEPATH = config['full_indexing_periods_input_file']
    ENCODING = config['encoding']
    DB_CONFIG = config['database']
  
    selective_indexing_periods = load_indexing_periods(SELECTIVE_INDEXING_PERIODS_FILEPATH, ENCODING, False)
    #full_indexing_periods = load_indexing_periods(FULL_INDEXING_PERIODS_FILEPATH, ENCODING, True)
    full_indexing_periods = {}
    journal_id_lookup = create_id_lookup(DB_CONFIG, SELECT_ALL_JOURNALS_SQL)

    table_data = _get_table_data(journal_id_lookup, selective_indexing_periods) + _get_table_data(journal_id_lookup, full_indexing_periods)
    table_data = sorted(table_data, key=lambda x: x['journal_id'])

    with closing(connect(**DB_CONFIG)) as conn:
        with closing(conn.cursor()) as cursor:    #pylint: disable=E1101
            for row in table_data:
                try:
                    cursor.execute(INSERT_SQL, row)    #pylint: disable=E1101
                except IntegrityError as ex:
                    print(ex.msg)
                    continue
        conn.commit()    #pylint: disable=E1101


def _get_table_data(journal_id_lookup, indexing_periods):
    table_data = []
    for nlm_id, periods in indexing_periods.items():
        if nlm_id in journal_id_lookup:
            for period in periods:
                period_dict = dict(period.items())
                period_dict['journal_id'] = journal_id_lookup[nlm_id]
                table_data.append(period_dict)
        else:
            print('Could not find journal with id {}'.format(nlm_id))
    return table_data


if __name__ == '__main__':
    config = get_config()
    run(config)

