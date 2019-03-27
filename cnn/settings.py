import json
from machine_settings import _MachineConfig
import os.path as os_path


ENCODING = 'utf8'


def get_config():
    config = Config()
    return config


class _ConfigBase:
    def __init__(self, parent):
        self._parent = parent
        machine_config = _MachineConfig()
        self._initialize(machine_config)

    def _initialize(self, machine_config):
        pass

    def __str__(self):
        dict = {}
        self._toJson(dict, self)
        return json.dumps(dict, indent=4)
            
    @classmethod
    def _toJson(cls, parent, obj):
        for attribute_name in dir(obj): 
            if not attribute_name.startswith('_'):
                attribute = getattr(obj, attribute_name)
                if isinstance(attribute, _ConfigBase):
                    child = {}
                    parent[attribute_name] = child
                    cls._toJson(child, attribute)
                else:
                    parent[attribute_name] = attribute 
        

class _CheckpointConfig(_ConfigBase):
    def _initialize(self, _):
  
        self.enabled = True
        self.weights_only = True
        self.dir = 'checkpoints'
        self.filename = 'best_model.hdf5'


class _CrossValidationConfig(_ConfigBase):
    def _initialize(self, machine_config):
        self.train_set_ids_path = os_path.join(machine_config.data_dir, 'preprocessed/cross-validation/group_1_train_set_1809-2018.txt')
        self.dev_set_ids_path = os_path.join(machine_config.data_dir, 'preprocessed/cross-validation/group_1_target_dev_set_2018-2018.txt')
        self.test_set_ids_path = os_path.join(machine_config.data_dir, 'preprocessed/cross-validation/group_1_reporting_test_set_2018-2018.txt')
        self.encoding = ENCODING
        self.train_limit = machine_config.train_limit
        self.dev_limit = machine_config.dev_limit
        self.test_limit = machine_config.test_limit


class _CsvLoggerConfig(_ConfigBase):
    def _initialize(self, _):

        self.dir = 'logs'
        self.filename  = 'logs.csv'
        self.best_epoch_filename = 'best_epoch_logs.txt'
        self.encoding = ENCODING


class _DatabaseConfig(_ConfigBase):
    def _initialize(self, machine_config):

        self.config = { 'user': '****',
                        'database': '****',
                        'password': '****', 
                        'host': machine_config.database_host,
                        'charset': 'utf8mb4', 
                        'collation': 'utf8mb4_unicode_ci', 
                        'use_unicode': True }


class _EarlyStoppingConfig(_ConfigBase):
    def _initialize(self, _):

        self.min_delta = 0.001
        self.patience = 2


class _ModelConfig(_ConfigBase):
    def _initialize(self, _):

        self.checkpoint = _CheckpointConfig(self)

        self.word_embedding_size = 300
        self.word_embedding_dropout_rate = 0.25
     
        self.conv_act = 'relu'
        self.num_conv_filter_sizes = 3
        self.min_conv_filter_size = 2
        self.conv_filter_size_step = 3
        self.total_conv_filters = 350
        self.num_pool_regions = 5

        self.num_journals = 30347
        self.journal_embedding_size = 50

        self.num_hidden_layers = 1
        self.hidden_layer_size = 3365
        self.hidden_layer_act = 'relu'
        self.inputs_dropout_rate = 0.0
        self.dropout_rate = 0.5

        self.output_layer_act = 'sigmoid'
        self.output_layer_size = self._pp_config.num_labels 
        
        self.init_threshold = 0.5
        self.init_learning_rate = 0.001

    @property
    def hidden_layer_sizes(self):
        return [self.hidden_layer_size]*self.num_hidden_layers

    @property
    def conv_filter_sizes(self):
        sizes = [self.min_conv_filter_size + self.conv_filter_size_step*idx for idx in range(self.num_conv_filter_sizes)]
        return sizes

    @property
    def conv_num_filters(self):
        num_filters = round(self.total_conv_filters / len(self.conv_filter_sizes))
        return num_filters

    @property
    def _pp_config(self):
        return self._parent.inputs.preprocessing

    @property
    def vocab_size(self):
        return self._pp_config.vocab_size

    @property
    def title_max_words(self):
        return self._pp_config.title_max_words

    @property
    def abstract_max_words(self):
        return self._pp_config.abstract_max_words

    @property
    def num_year_completed_time_periods(self):
        return self._pp_config.num_year_completed_time_periods

    @property
    def num_pub_year_time_periods(self):
        return self._pp_config.num_pub_year_time_periods


class _PreprocessingConfig(_ConfigBase):
    def _initialize(self, machine_config):

        self.word_index_lookup_path = os_path.join(machine_config.data_dir, 'preprocessed/vocab/cross_val_group_1_word_index_lookup.pkl') # indices start from 2
        self.unknown_index = 1
        self.padding_index = 0
        self.title_max_words = 64
        self.abstract_max_words = 448
        self.num_labels = 1  
        self.vocab_size = 400000
        self.min_year_completed= 1965 
        self.max_year_completed = 2018       
        self.num_year_completed_time_periods = 1 + self.max_year_completed - self.min_year_completed
        self.min_pub_year = 1809 
        self.max_pub_year = 2018    
        self.num_pub_year_time_periods = 1 + self.max_pub_year - self.min_pub_year

      
class _ProcessingConfig(_ConfigBase):
    def _initialize(self, machine_config):
              
        self.run_on_cpu = machine_config.run_on_cpu                                         
        self.use_multiprocessing = machine_config.use_multiprocessing                                
        self.workers = machine_config.workers                                                
        self.max_queue_size = machine_config.max_queue_size


class _ReduceLearningRateConfig(_ConfigBase):
    def _initialize(self, _):

        self.factor = 0.33
        self.patience = 1
        self.min_delta = 0.001


class _RestoreConfig(_ConfigBase):
     def _initialize(self, machine_config):
        super()._initialize(machine_config)
         
        self.sub_dir = '****'
        self.model_json_filename = 'model.json'
        self.encoding = ENCODING
        self.model_checkpoint_dir = 'checkpoints'
        self.model_checkpoint_filename = 'best_model.hdf5'
        self.weights_only_checkpoint = True
        self.threshold = 0.5
        self.learning_rate = 0.001
        

class _ResumeConfig(_RestoreConfig):
     def _initialize(self, machine_config):
        super()._initialize(machine_config)

        self.enabled = False
        self.resume_checkpoint_filename = 'best_model_resume.hdf5'
        self.resume_logger_filename  = 'logs_resume.csv'


class _SaveConfig(_ConfigBase):
    def _initialize(self, _):

        self.settings_filename = 'settings.json'
        self.model_json_filename = 'model.json'
        self.encoding = ENCODING
        self.model_img_filename = 'model.png'


class _TensorboardConfig(_ConfigBase):
    def _initialize(self, _):

        self.enabled = False
        self.dir = 'logs'
        self.write_graph = True


class _EvaluateConfig(_RestoreConfig, _ProcessingConfig):
    def _initialize(self, machine_config):
        super()._initialize(machine_config)

        self.results_filename = 'eval-result.txt'
        self.encoding = ENCODING
        self.batch_size = 128
        self.limit = 1000000000


class _PredictConfig(_RestoreConfig, _ProcessingConfig):
    def _initialize(self, machine_config):
        super()._initialize(machine_config)

        self.pmids_filepath = '../pipeline_validation_set.json'
        self.results_dir = 'predictions'
        self.results_filename = 'predictions.csv'
        self.dereferenced_filename = 'dereferenced_predictions.csv'
        self.metrics_filename_template = 'metrics{}.csv'
        self.journal_groups_filepath = os_path.join(machine_config.data_dir, 'preprocessed/selective-indexing/selectively_indexed_journal_groups.csv') 
        self.encoding = ENCODING
        self.delimiter = ','
        self.batch_size = 128
        self.limit = 1000000000
    
 
class _InputsConfig(_ConfigBase):
    def _initialize(self, _):

        self.preprocessing = _PreprocessingConfig(self)


class _OptimizeFscoreThresholdConfig(_ProcessingConfig):
    def _initialize(self, machine_config):
        super()._initialize(machine_config)
        
        self.enabled = True
        self.batch_size = 128
        self.limit = 1000000000
        self.metric_name = 'fscore'
        self.alpha = 0.005
        self.k = 3


class _TrainingConfig(_ProcessingConfig):
    def _initialize(self, machine_config):
        super()._initialize(machine_config)

        self.batch_size = 128
        self.initial_epoch = 0
        self.max_epochs = 500
        self.train_limit = 1000000000
        self.dev_limit = 1000000000
        self.monitor_metric = 'val_fscore'
        self.monitor_mode = 'max'
        self.save_config = _SaveConfig(self)
        self.optimize_fscore_threshold = _OptimizeFscoreThresholdConfig(self)
        self.reduce_learning_rate = _ReduceLearningRateConfig(self)
        self.early_stopping = _EarlyStoppingConfig(self)
        self.tensorboard = _TensorboardConfig(self)
        self.csv_logger = _CsvLoggerConfig(self)
        self.resume = _ResumeConfig(self)


class Config(_ConfigBase):
    def __init__(self):
        super().__init__(self)

    def _initialize(self, machine_config):

        self.root_dir = machine_config.runs_dir
        self.data_dir = machine_config.data_dir
        self.inputs = _InputsConfig(self)
        self.model = _ModelConfig(self)
        self.cross_val = _CrossValidationConfig(self)
        self.train = _TrainingConfig(self)
        self.eval = _EvaluateConfig(self)
        self.pred = _PredictConfig(self)
        self.database = _DatabaseConfig(self)