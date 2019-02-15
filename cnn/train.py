import env_variables
from contextlib import ExitStack
from data_helper import DatabaseGenerator, load_cross_validation_ids, load_pickled_object, set_vocab_size
from model import Model
import os.path as os_path
from settings import get_config
import tensorflow as tf
from time import time


def train_model(config, output_dir):
    
    with ExitStack() as stack:

        # Load config
        db_config = config.database
        pp_config = config.inputs.preprocessing
        ofs_config = config.train.optimize_fscore_threshold
        model_config = config.model
        resume_config = config.train.resume

        # Get inputs
        train_set_ids, dev_set_ids = load_cross_validation_ids(config.cross_val)
        word_index_lookup = load_pickled_object(pp_config.word_index_lookup_path)
        word_index_lookup = set_vocab_size(word_index_lookup, pp_config.vocab_size)
        train_gen = DatabaseGenerator(db_config, pp_config, word_index_lookup, train_set_ids, config.train.batch_size, config.train.train_limit)
        dev_gen =   DatabaseGenerator(db_config, pp_config, word_index_lookup, dev_set_ids, config.train.batch_size, config.train.dev_limit)
        opt_gen =   DatabaseGenerator(db_config, pp_config, word_index_lookup, dev_set_ids, ofs_config.batch_size, ofs_config.limit)

        # Configure run on cpu
        if config.train.run_on_cpu:
            stack.enter_context(tf.device('/cpu:0'))

        # Create/load model
        model = Model()
        if resume_config.enabled:
            model.restore(resume_config, output_dir)
        else:
            model.build(model_config)

        # Fit
        best_epoch_logs = model.fit(config, train_gen, dev_gen, opt_gen, output_dir)

    return best_epoch_logs


if __name__ == '__main__':
    config = get_config()
    resume_config = config.train.resume
    if resume_config.enabled:
        subdir = resume_config.sub_dir
    else:
        subdir = str(int(time()))
    output_dir = os_path.join(config.root_dir, subdir)
    train_model(config, output_dir)
    
 
    




