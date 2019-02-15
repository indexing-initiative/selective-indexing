import env_variables
from contextlib import ExitStack
from data_helper import DatabaseGenerator, load_db_ids, load_pickled_object, save_delimited_data, set_vocab_size
from model import Model
from os import mkdir
import os.path as os_path
from scripts import dereference_predictions_file, save_performance_metrics
from settings import get_config
import tensorflow as tf

import json
from contextlib import closing
from mysql.connector import connect

def create_id_lookup(db_config, sql):
    lookup = {}
    with closing(connect(**db_config)) as conn:
        with closing(conn.cursor()) as cursor:    #pylint: disable=E1101
            cursor.execute(sql)                   #pylint: disable=E1101
            for row in cursor.fetchall():         #pylint: disable=E1101
                id, ui = row
                lookup[ui] = id
    return lookup


def predict(config, input_dir):
    with ExitStack() as stack:

        cross_val_cfg = config.cross_val
        preprocessing_cfg = config.inputs.preprocessing
        database_cfg = config.database
        pred_cfg = config.pred
    
        #test_set_ids = load_db_ids(cross_val_cfg.test_set_ids_path, cross_val_cfg.encoding, cross_val_cfg.test_limit)
        pmids = [doc['pmid'] for doc in json.load(open('/****/reporting_test_set.json', 'rt', encoding='utf8'))]
        id_lookup = create_id_lookup(database_cfg.config, 'SELECT id, pmid from articles')
        test_set_ids = [id_lookup[pmid] for pmid in pmids]

        word_index_lookup = load_pickled_object(preprocessing_cfg.word_index_lookup_path)
        word_index_lookup = set_vocab_size(word_index_lookup, preprocessing_cfg.vocab_size)
        test_gen = DatabaseGenerator(database_cfg, preprocessing_cfg, word_index_lookup, test_set_ids, pred_cfg.batch_size, pred_cfg.limit)
    
        if pred_cfg.run_on_cpu:
            stack.enter_context(tf.device('/cpu:0'))

        model = Model()
        model.restore(pred_cfg, input_dir)
        predictions = model.predict(pred_cfg, test_gen)

        _save_predictions(pred_cfg, predictions, input_dir)
        _dereference_prediction_file(pred_cfg, database_cfg.config, input_dir)
        metric_groupings = {
            '': [-1],
            '-by-pub-year': [1],
            '-by-journal': [3],
            '-by-journal-group': [4],
            '-by-journal-group-pub-year': [4,1],
        }
        for grouping_name, group_index in metric_groupings.items():
            _save_performance_metrics(pred_cfg, input_dir, grouping_name, group_index)
        
    return predictions


def _dereference_prediction_file(pred_cfg, db_config, input_dir):
    results_dir = os_path.join(input_dir, pred_cfg.results_dir)
    predictions_filepath = os_path.join(results_dir, pred_cfg.results_filename)
    dereferenced_predictions_filepath = os_path.join(results_dir, pred_cfg.dereferenced_filename)
    dereference_predictions_file.run(predictions_filepath, pred_cfg.journal_groups_filepath, db_config, dereferenced_predictions_filepath, pred_cfg.encoding, pred_cfg.delimiter)


def _save_performance_metrics(pred_cfg, input_dir, grouping_name, group_index):
    results_dir = os_path.join(input_dir, pred_cfg.results_dir)
    dereferenced_predictions_filepath = os_path.join(results_dir, pred_cfg.dereferenced_filename)
    metrics_filename = pred_cfg.metrics_filename_template.format(grouping_name)
    metrics_filepath = os_path.join(results_dir, metrics_filename)
    save_performance_metrics.run(dereferenced_predictions_filepath, pred_cfg.encoding, pred_cfg.delimiter, group_index, metrics_filepath, pred_cfg.threshold)


def _save_predictions(pred_cfg, predictions, input_dir):
    results_dir = os_path.join(input_dir, pred_cfg.results_dir)
    if not os_path.isdir(results_dir): 
        mkdir(results_dir)
    results_filepath = os_path.join(results_dir, pred_cfg.results_filename)
    save_delimited_data(results_filepath, pred_cfg.encoding, pred_cfg.delimiter, predictions)


if __name__ == '__main__':
    config = get_config()
    input_dir = os_path.join(config.root_dir, config.pred.sub_dir)
    predict(config, input_dir)
