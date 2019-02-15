from contextlib import closing
from helper import load_delimited_data, load_ids_from_file, write_ids_to_file
from mysql.connector import connect
from settings import get_config


GET_ARTICLE_NLMID_SQL = 'SELECT nlmid FROM articles AS a, journals AS j WHERE a.journal_id = j.id AND a.id = %(id)s'


def run(config):
    DB_CONFIG = config['database']
    ENCODING = config['encoding']
    TEST_SET_IDS_FILEPATH = config['test_set_ids_file']
    REPORTING_JOURNALS_FILEPATH = config['reporting_journals_file']
    REPORTING_TEST_SET_IDS_FILEPATH = config['reporting_test_set_ids_file']

    reporting_nlmids = [row[0] for row in load_delimited_data(REPORTING_JOURNALS_FILEPATH, ENCODING, ',')]
    test_set_ids = load_ids_from_file(TEST_SET_IDS_FILEPATH, ENCODING)
    reporting_ids = []
    with closing(connect(** DB_CONFIG )) as conn:
        with closing(conn.cursor()) as cursor:                    #pylint: disable=E1101
            for id in test_set_ids:
                cursor.execute(GET_ARTICLE_NLMID_SQL, {'id': id}) #pylint: disable=E1101
                nlmid = cursor.fetchone()[0]                      #pylint: disable=E1101
                if nlmid in reporting_nlmids:
                    reporting_ids.append(id)

    write_ids_to_file(REPORTING_TEST_SET_IDS_FILEPATH, ENCODING, reporting_ids)


if __name__ == '__main__':
    config = get_config()
    run(config)


