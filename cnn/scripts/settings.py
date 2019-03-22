def get_config():

    downloaded_data_dir =  '{root_dir}/downloaded-data'
    extracted_data_dir =  '{root_dir}/extracted-data'
    preprocessed_dir = '{root_dir}/preprocessed'
    selective_indexing_dir = preprocessed_dir + '/selective-indexing'
    cross_val_dir =  preprocessed_dir + '/cross-validation'

    config = {
        # Machine specific
        'root_dir' : '****',
        # Year specific
        'data_filename_template' : 'pubmed18n{0:04d}',
        'url_template' : 'https://mbr.nlm.nih.gov/Download/Baselines/2018/{data_filename_template}.xml.gz',
        'start_data_file_num' : 1, 
        'end_download_data_file_num': 928,
        'end_data_file_num': 1250,
        'medline_citation_node_path': 'PubmedArticle/MedlineCitation',
        
        # Shared
        'encoding' : 'utf8',

        'data_filepath_template': downloaded_data_dir + '/{data_filename_template}.xml.gz',

        'serials_file': '{root_dir}/lsi2018.xml',
        'reporting_journals_file': selective_indexing_dir + '/selectively-indexed-journals-of-interest.csv',

        'selective_indexing_periods_output_file': selective_indexing_dir + '/selective_indexing_periods_output.txt',
        'full_indexing_periods_output_file': selective_indexing_dir + '/full_indexing_periods_output.txt',
        'selective_indexing_periods_input_file': selective_indexing_dir + '/selective_indexing_periods_input.csv',
        'full_indexing_periods_input_file': selective_indexing_dir + '/full_indexing_periods_input.csv',

        'json_filepath_template': extracted_data_dir + '/{data_filename_template}.json.gz',
        'extract_data_log_file':  extracted_data_dir + '/extract_data_log.txt',

        'journals_file': '{root_dir}/J_Medline.txt',
        'problematic_journals_file': selective_indexing_dir + '/problematic_journals.csv',
      
        'database': { 'host': '****', 'user': '****', 'database': 'selective_indexing_2018', 'password': '****', 'charset': 'utf8mb4', 'collation': 'utf8mb4_unicode_ci', 'use_unicode': True },
        'max_abs_len': 13000,
        'load_data_log_file': extracted_data_dir + '/load_data_log.txt',

        # Cross val
        'test_set_start_year' : 2018, 
        'test_set_end_year' : 2018, 
        'target_dev_set_start_year' : 2018,
        'target_dev_set_end_year' : 2018, 
        'dev_set_start_year' : 1809,
        'dev_set_end_year' : 2018, 
        'train_set_start_year' : 1809, 
        'train_set_end_year' : 2018, 

        'test_set_size' : 30000,
        'target_dev_set_size' : 30000,
        'dev_set_size' : 15000,
        'train_set_size' : 1000000000,
   
        'cross_val_use_existing' : False,
        'cross_val_group_num' : 1,
       
        'test_set_ids_file' : cross_val_dir + '/group_{cross_val_group_num}_test_set_{test_set_start_year:04d}-{test_set_end_year:04d}.txt',
        'reporting_test_set_ids_file' : cross_val_dir + '/group_{cross_val_group_num}_reporting_test_set_{test_set_start_year:04d}-{test_set_end_year:04d}.txt', 
        'target_dev_set_ids_file' : cross_val_dir + '/group_{cross_val_group_num}_target_dev_set_{target_dev_set_start_year:04d}-{target_dev_set_end_year:04d}.txt',
        'dev_set_ids_file' : cross_val_dir + '/group_{cross_val_group_num}_dev_set_{dev_set_start_year:04d}-{dev_set_end_year:04d}.txt',
        'cleaned_dev_set_ids_file' : cross_val_dir + '/group_{cross_val_group_num}_cleaned_dev_set_{dev_set_start_year:04d}-{dev_set_end_year:04d}.txt',
        'train_set_ids_file' : cross_val_dir + '/group_{cross_val_group_num}_train_set_{train_set_start_year:04d}-{train_set_end_year:04d}.txt',
        'cleaned_train_set_ids_file' : cross_val_dir + '/group_{cross_val_group_num}_cleaned_train_set_{train_set_start_year:04d}-{train_set_end_year:04d}.txt',

        'word_index_lookup_file' : preprocessed_dir + '/vocab/cross_val_group_{cross_val_group_num}_word_index_lookup.pkl',
        'journal_groups_file': selective_indexing_dir + '/selectively_indexed_journal_groups.csv'
        }

    # Preprocessing
    _format(config, 'url_template')
    _format(config, 'data_filepath_template')
    _format(config, 'serials_file')
    _format(config, 'json_filepath_template')
    _format(config, 'extract_data_log_file')
    _format(config, 'reporting_journals_file')
    
    _format(config, 'selective_indexing_periods_output_file')
    _format(config, 'full_indexing_periods_output_file')
    _format(config, 'selective_indexing_periods_input_file')
    _format(config, 'full_indexing_periods_input_file')

    _format(config, 'journals_file')
    _format(config, 'problematic_journals_file')
    _format(config, 'load_data_log_file')

    _format(config, 'test_set_ids_file')
    _format(config, 'reporting_test_set_ids_file')
    _format(config, 'target_dev_set_ids_file')
    _format(config, 'dev_set_ids_file')
    _format(config, 'cleaned_dev_set_ids_file')
    _format(config, 'train_set_ids_file')
    _format(config, 'cleaned_train_set_ids_file')

    _format(config, 'word_index_lookup_file')
    _format(config, 'journal_groups_file')
 
    return config

def _format(config, key):
    config[key] = config[key].format(**config)