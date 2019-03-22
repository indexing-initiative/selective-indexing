from contextlib import closing
from helper import create_journal_group_name_lookup, create_id_lookup
from mysql.connector import connect
from settings import get_config


SELECT_ALL_JOURNAL_GROUPS_SQL = 'SELECT id, name FROM journal_groups'
INSERT_JOURNAL_SQL = 'INSERT INTO journals (id, nlmid, medline_ta, group_id) VALUES (%(id)s, %(nlmid)s, %(medline_ta)s, %(group_id)s)'


def run(config):
    JOURNALS_FILEPATH = config['journals_file']
    ENCODING = config['encoding']
    JOURNAL_GROUPS_FILEPATH = config['journal_groups_file']
    DB_CONFIG = config['database']

    journal_data_list = _get_journal_data_list(JOURNALS_FILEPATH, ENCODING)
    journal_data_list = _append_journal_groups(JOURNAL_GROUPS_FILEPATH, ENCODING, journal_data_list)
    _populate_journals_table(DB_CONFIG, journal_data_list)


def _append_journal_groups(journal_groups_filepath, encoding, journal_data_list):
    journal_group_name_lookup = create_journal_group_name_lookup(journal_groups_filepath, encoding, ',')

    updated_journal_data_list = []
    for journal_data in journal_data_list:
        nlmid = journal_data['nlmid'] 
        group_name = 'NotAssigned' 
        if nlmid in journal_group_name_lookup:
            group_name = journal_group_name_lookup[nlmid]
        journal_data['group_name'] = group_name
        updated_journal_data_list.append(dict(journal_data.items()))

    return updated_journal_data_list


def _get_journal_data_list(journals_filepath, encoding):
    journal_data_list = []
    with open(journals_filepath, 'rt', encoding=encoding) as journals_file:
        lines = journals_file.readlines()
    line_count = len(lines)
    lines_per_journal = 8
    journal_count = line_count // lines_per_journal
    for idx in range(journal_count):
        start_line = lines_per_journal*idx
        #jrid =   int(lines[start_line + 1].strip()[6:].strip())
        nlmid =      lines[start_line + 7].strip()[7:].strip()
        medline_ta = lines[start_line + 3].strip()[9:].strip()
        journal_data = {'nlmid': nlmid, 'medline_ta': medline_ta}
        journal_data_list.append(journal_data)
    return journal_data_list
            
    
def _populate_journals_table(db_config, journal_data_list):
    journal_group_id_lookup = create_id_lookup(db_config, SELECT_ALL_JOURNAL_GROUPS_SQL)
    with closing(connect(**db_config)) as conn:
        with closing(conn.cursor()) as cursor:     #pylint: disable=E1101
            for index, journal_data in enumerate(journal_data_list):
                id = index + 1
                nlmid, medline_ta, group_name = journal_data['nlmid'], journal_data['medline_ta'], journal_data['group_name']
                group_id = journal_group_id_lookup[group_name]
                cursor.execute(INSERT_JOURNAL_SQL, { 'id': id, 'nlmid': nlmid, 'medline_ta' : medline_ta, 'group_id': group_id })     #pylint: disable=E1101
        conn.commit()     #pylint: disable=E1101


if __name__ == '__main__':
    config = get_config()
    run(config)

