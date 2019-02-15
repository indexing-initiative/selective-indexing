from urllib.request import urlretrieve
from settings import get_config


def run(config):
    URL_TEMPLATE = config['url_template']
    START_DATA_FILE_NUM = config['start_data_file_num']
    END_DATA_FILE_NUM = config['end_data_file_num']
    DATA_FILEPATH_TEMPLATE = config['data_filepath_template']

    for file_num in range(START_DATA_FILE_NUM, END_DATA_FILE_NUM + 1):
        url = URL_TEMPLATE.format(file_num)
        file = DATA_FILEPATH_TEMPLATE.format(file_num)
        urlretrieve(url, file)


if __name__ == '__main__':
    config = get_config()
    run(config)