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
    def __init__(self , data_path, joint_file, point_num, inputtype = "position", outputtype = "direct"):
        self.data_path = data_path
        self.data_files = get_all_files(data_path)
        self.inputtype = inputtype
        self.outputtype = outputtype
        self.point_num = point_num
        print("The number of data files is %d" % len(self.data_files))

        joint_info = json.load(open(joint_file))
        self.joints = joint_info['joints']
        self.parent = joint_info['parent']
        self.skel_offset = joint_info['skel_offset']
        self.sct = SkeletonCoordinatesTransform(self.joints, self.parent, self.skel_offset)
        self.cache_scheduler = CacheSchedule(data_path, self.data_files)
        self.joint2id = {}
        for id, joint in enumerate(self.joints):
            self.joint2id[joint] = id



    '''
    Convert list joint_num * [x,y,z,w] * point_num * [ [n, v], k ]
    to point_num * joint_num * [xv, yv,zv,wv]
    '''
    def get_local_rotation(self,data):
        data = np.array(data)
        assert data.shape[2] == self.point_num
        assert data.shape[0] == len(self.joints)
        assert data.shape[1] == 4
        local_rotations = np.zeros([self.point_num,len(self.joints),4])
        for time in range(self.point_num):
            for joint in range(len(self.joints)):
                local_rotations[time][joint] = [x[time][0][1] for x in data[joint]]
        return torch.tensor(local_rotations,dtype = torch.double)

    def get_inputs(self, data):
        if self.inputtype == "position":
            pass

    def __getitem__(self, index):
        data = self.cache_scheduler.load(index)
        local_rotations = self.get_local_rotation(data["rotations"])
        #inputs = get_inputs(data)
        return local_rotations
        
    def __len__(self):
        return len(self.data_files)

if __name__ == '__main__':
    motiondataset = MotionDataset("data_example\data\\","data_example\joint_info.json",16)
    x = motiondataset[0]
    import ipdb;ipdb.set_trace()