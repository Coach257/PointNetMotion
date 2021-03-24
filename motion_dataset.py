import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from utils.data_util import get_all_files
from utils.cache_scheduler import CacheSchedule

class MotionDataset(Dataset):
    def __init__(self , data_path):
        self.data_path = data_path
        self.data_files = get_all_files(data_path)
        print("The number of data files is %d" % len(self.data_files))

        self.cache_scheduler = CacheSchedule(data_path, self.data_files)
        