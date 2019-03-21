from collections import defaultdict
from contextlib import closing
from datetime import datetime as dt
from itertools import islice
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import Sequence
import math
from mysql.connector import connect
import numpy as np
from nltk.tokenize import word_tokenize
from operator import itemgetter
import pickle
from random import sample


SELECT_ARTICLE_DATA_SQL_TEMPLATE = 'SELECT id, title, abstract, pub_year, date_completed, journal_id, is_indexed FROM articles where id IN ({})'


def _word_tokenize(text):
    text = text.lower()
    tokens = word_tokenize(text)
    return tokens


def create_word_index_lookup(db_conn, db_ids, batch_size = 1000, tokenizer = _word_tokenize, verbose=False):       
    sorted_ids = sorted(db_ids)
    num_ids = len(sorted_ids)
    num_batches = int(math.ceil(num_ids/batch_size))

    token_counts = defaultdict(int)
    for idx in range(num_batches):
        batch_start = idx * batch_size
        batch_end = (idx + 1) * batch_size
        batch_ids = sorted_ids[batch_start:batch_end]
        title, abstract, _, _, _, _ = _get_batch_data(db_conn, batch_ids)
        for idx in range(len(title)):
            title_tokens = tokenizer(title[idx])
            abstract_tokens = tokenizer(abstract[idx])
            tokens = title_tokens + abstract_tokens
            for token in tokens:
                token_counts[token] += 1
        if verbose:
            print(batch_end)

    sorted_items = sorted(token_counts.items(), key=itemgetter(1), reverse=True)
    lookup = { item[0]: index + 2 for index, item in enumerate(sorted_items) }
    return lookup
    

def load_cross_validation_ids(config):
    train_set_ids = load_db_ids(config.train_set_ids_path, config.encoding, config.train_limit)
    dev_set_ids =  load_db_ids(config.dev_set_ids_path, config.encoding, config.dev_limit)
    return train_set_ids, dev_set_ids


def load_db_ids(path, encoding, limit = 1000000000):
    with open(path, 'rt', encoding=encoding) as file:
        db_ids = [int(line.strip()) for line in islice(file, limit)]
    return db_ids


def load_pickled_object(path):
    loaded_object = pickle.load(open(path, 'rb'))
    return loaded_object


def save_delimited_data(path, encoding, delimiter, data):
    with open(path, 'wt', encoding=encoding) as file:
        for data_row in data:
            line = delimiter.join([str(data_item) for data_item in data_row]) + '\n'
            file.write(line)


def set_vocab_size(word_index_lookup, vocab_size):
    sorted_items = sorted(word_index_lookup.items(), key=itemgetter(1))
    word_index_lookup_size = vocab_size - 2 # minus unknown/padding
    modified_lookup = dict(sorted_items[:word_index_lookup_size])
    assert(len(modified_lookup) == word_index_lookup_size)
    return modified_lookup


def _get_batch_data(db_conn, db_ids):
    sql = SELECT_ARTICLE_DATA_SQL_TEMPLATE.format(','.join([str(db_id) for db_id in db_ids]))
    inputs_lookup = {}
    with closing(db_conn.cursor()) as cursor:
        cursor.execute(sql)    #pylint: disable=E1101
        for row in cursor.fetchall():    #pylint: disable=E1101
            id, title, abstract, pub_year, date_completed, journal_id, is_indexed = row
            year_completed = date_completed.year
            if not journal_id:
                journal_id = 0
            inputs_lookup[id] = (title, abstract, pub_year, year_completed, journal_id, is_indexed)
    ordered_inputs = [inputs_lookup[db_id] for db_id in db_ids]
    return zip(*ordered_inputs)


class DatabaseGenerator(Sequence):

    def __init__(self, db_config, pp_config, word_index_lookup, db_ids, batch_size, max_examples = 1000000000, tokenizer=_word_tokenize):
        self._db_config = db_config
        self._pp_config = pp_config
        self._word_index_lookup = word_index_lookup
        self._db_ids = db_ids
        self._batch_size = batch_size 
        self._num_examples = min(len(db_ids), max_examples)
        self._tokenizer = tokenizer
   
    def __len__(self):
        length = int(math.ceil(self._num_examples/self._batch_size))
        return length

    def __getitem__(self, idx):
        with closing(connect(**self._db_config.config)) as db_conn:
            batch_start = idx * self._batch_size
            batch_end = (idx + 1) * self._batch_size
            batch_ids = self._db_ids[batch_start:batch_end]

            article_ids = np.array(batch_ids, dtype=np.int32).reshape(-1, 1)

            title, abstract, pub_year, year_completed, journal_id, is_indexed = _get_batch_data(db_conn, batch_ids)
            title_input = self._vectorize_batch_text(title, self._pp_config.title_max_words)
            abstract_input = self._vectorize_batch_text(abstract, self._pp_config.abstract_max_words)

            pub_year = np.array(pub_year, dtype=np.uint16).reshape(-1, 1)
            pub_year_indices = pub_year - self._pp_config.min_pub_year
            pub_year_input = self._to_time_period_input(pub_year_indices, self._pp_config.num_pub_year_time_periods)

            year_completed = np.array(year_completed, dtype=np.uint16).reshape(-1, 1)
            year_completed_indices = year_completed - self._pp_config.min_year_completed
            year_completed_input = self._to_time_period_input(year_completed_indices, self._pp_config.num_year_completed_time_periods)

            journal_input = np.array(journal_id, dtype=np.uint16).reshape(-1, 1)

            batch_x = { 'article_ids': article_ids, 'title_input': title_input, 'abstract_input': abstract_input, 'pub_year_input': pub_year_input, 'year_completed_input': year_completed_input, 'journal_input': journal_input}
        
            is_indexed = np.array(is_indexed, dtype='float32').reshape(-1, 1)

            batch_y = is_indexed
          
            return batch_x, batch_y

    def _to_categorical(self, ids, num_labels):
        count = ids.shape[0]
        max_labels = ids.shape[1]
        batch_indices = np.zeros([max_labels, count], np.uint16)
        batch_indices[np.arange(max_labels)] = np.arange(count)
        batch_indices = batch_indices.T
        one_hot = np.zeros([count, num_labels + 1], np.uint8)
        one_hot[batch_indices, ids] = 1
        one_hot = one_hot[:, 1:]
        return one_hot

    def _to_time_period_input(self, year_indices, num_time_periods):
        batch_size = year_indices.shape[0]
        batch_indices = np.zeros([batch_size, num_time_periods], np.uint16)
        batch_indices[np.arange(batch_size)] = np.arange(num_time_periods)
        year_indices_rep = np.repeat(year_indices, num_time_periods, axis=1)
        time_period_input = batch_indices <= year_indices_rep
        time_period_input = time_period_input.astype(np.uint8)
        return time_period_input

    def _vectorize_batch_text(self, batch_text, max_words):
        batch_words = [self._tokenizer(text) for text in batch_text]
        batch_word_indices = [[self._word_to_index(word) for word in words] for words in batch_words]
        vectorized_text = pad_sequences(batch_word_indices, maxlen=max_words, dtype='int32', padding='post', truncating='post', value=self._pp_config.padding_index)
        return vectorized_text

    def _word_to_index(self, word):
        index = self._word_index_lookup[word] if word in self._word_index_lookup else self._pp_config.unknown_index
        return index