import env_variables
from contextlib import ExitStack
from data_helper import DatabaseGenerator, load_db_ids, load_pickled_object, set_vocab_size
from model import Model
import os.path as os_path
from settings import get_config
import tensorflow as tf


def eval_model(config, input_dir):
    with ExitStack() as stack:

        cross_val_cfg = config.cross_val
        preprocessing_cfg = config.inputs.preprocessing
        database_cfg = config.database
        eval_cfg = config.eval
    
        test_set_ids = load_db_ids(cross_val_cfg.test_set_ids_path, cross_val_cfg.encoding, cross_val_cfg.test_limit)
        word_index_lookup = load_pickled_object(preprocessing_cfg.word_index_lookup_path)
        word_index_lookup = set_vocab_size(word_index_lookup, preprocessing_cfg.vocab_size)
        test_gen = DatabaseGenerator(database_cfg, preprocessing_cfg, word_index_lookup, test_set_ids, eval_cfg.batch_size, eval_cfg.limit)
    
        if eval_cfg.run_on_cpu:
            stack.enter_context(tf.device('/cpu:0'))

        model = Model()
        model.restore(eval_cfg, input_dir)
        eval_result = model.evaluate(eval_cfg, test_gen, input_dir)
        
    return eval_result


if __name__ == '__main__':
    config = get_config()
    input_dir = os_path.join(config.root_dir, config.eval.sub_dir)
    eval_model(config, input_dir)