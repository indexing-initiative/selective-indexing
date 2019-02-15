from contextlib import closing
from mysql.connector import connect
from settings import get_config

INSERT_SQL = 'INSERT INTO ref_types (id, name) VALUES (%(id)s, %(name)s)'

REF_TYPES = ('AssociatedDataset',
             'AssociatedPublication',
             'CommentOn',
             'CommentIn',
             'ErratumIn',
             'ErratumFor',
             'ExpressionOfConcernIn',
             'ExpressionOfConcernFor',
             'RepublishedFrom',
             'RepublishedIn',
             'RetractionOf',
             'RetractionIn',
             'UpdateIn',
             'UpdateOf',
             'SummaryForPatientsIn',
             'OriginalReportIn',
             'ReprintOf',
             'ReprintIn')


def run(config):
    DB_CONFIG = config['database']
    with closing(connect(**DB_CONFIG)) as conn:
        with closing(conn.cursor()) as cursor:    #pylint: disable=E1101
            for index, name in enumerate(REF_TYPES):
                id = index + 1
                cursor.execute(INSERT_SQL, { 'id': id, 'name': name })    #pylint: disable=E1101
        conn.commit()    #pylint: disable=E1101


if __name__ == '__main__':
    config = get_config()
    run(config)

