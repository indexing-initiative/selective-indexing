class _MachineConfig:
    def __init__(self):
     
        self.data_dir = '****'
        self.database_host = '****'
        self.train_limit = 1000000000
        self.dev_limit = 1000000000
        self.test_limit = 1000000000

        self.runs_dir = '****'
        self.run_on_cpu = False
        self.use_multiprocessing = True 
        self.workers = 3               
        self.max_queue_size = 10