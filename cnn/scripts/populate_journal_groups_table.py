from contextlib import closing
from mysql.connector import connect
from settings import get_config

INSERT_JOURNAL_GROUP_SQL = 'INSERT INTO journal_groups (id, name) VALUES (%(id)s, %(name)s)'

JOURNAL_GROUPS = (
    'Biotech',
    'Chemistry',
    'History',
    'Jurisprudence',
    'Science',
    'NotAssigned',
)

def run(config):
    DB_CONFIG = config['database']
    with closing(connect(**DB_CONFIG)) as conn:
        with closing(conn.cursor()) as cursor:    #pylint: disable=E1101
            for index, name in enumerate(JOURNAL_GROUPS):
                id = index + 1
                cursor.execute(INSERT_JOURNAL_GROUP_SQL, { 'id': id, 'name': name })    #pylint: disable=E1101
        conn.commit()    #pylint: disable=E1101


if __name__ == '__main__':
    config = get_config()
    run(config)

