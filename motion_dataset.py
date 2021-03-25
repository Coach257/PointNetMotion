import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from utils.data_util import get_all_files
from utils.cache_scheduler import CacheSchedule
from utils.coordinates_transform import SkeletonCoordinatesTransform

__inputtype__ = ["position","gradient"]
__outputtype__ = ["direct","deviation"]
class MotionDataset(Dataset):
    def __init__(self , data_path, joint_file, inputtype = "position", outputtype = "direct "):
        self.data_path = data_path
        self.data_files = get_all_files(data_path)
        self.type = type
        print("The number of data files is %d" % len(self.data_files))

        joint_info = json.load(open(joint_file))
        self.joints = joint_info['joints']
        self.parent = joint_info['parent']
        self.skel_offset = joint_info['skel_offset']
        self.sct = SkeletonCoordinatesTransform(self.joints, self.parent, self.skel_offset)
        self.cache_scheduler = CacheSchedule(data_path, self.data_files)
        for id, joint in enumerate(self.joints):
            self.joint2id[joint] = id
                
    def __getitem__(self, index):

    def __len__(self):
        return len(self.data_files)
