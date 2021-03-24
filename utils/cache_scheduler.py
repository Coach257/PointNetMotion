import os
import json
import numpy as np

"""
Schedule cache for data loading
"""
class CacheSchedule(object):
    def __init__(self, data_path, files):
        self.cache_size = cache_size
        self.cache = [json.load(open(os.path.join(data_path, f), 'rb')) for f in files]

    def load(self, index):
        return self.cache[index]