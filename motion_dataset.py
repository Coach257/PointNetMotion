import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from utils.data_util import get_all_files
from utils.cache_scheduler import CacheSchedule

__inputtype__ = ["position","gradient"]
__outputtype__ = ["direct","deviation"]
class MotionDataset(Dataset):
    def __init__(self , data_path, joint_file, point_num, inputtype = "position", outputtype = "direct"):
        assert inputtype in __inputtype__
        assert outputtype in __outputtype__
        self.data_path = data_path
        self.data_files = get_all_files(data_path)
        self.inputtype = inputtype
        self.outputtype = outputtype
        self.point_num = point_num
        print("The number of data files is %d" % len(self.data_files))
        self.cache_scheduler = CacheSchedule(data_path, self.data_files)

        joint_info = json.load(open(joint_file))
        self.joints = joint_info['joints']


    def normalize(self,data):
        data = torch.tensor(data , dtype = torch.float)
        max_len = data[0][0][-1][0]
        data[:][:][:] = data[:][:][:]/torch.tensor([max_len,1,1])
        return data
    '''
    Convert list: 
    joint_num * [x,y,z,w,px,py,pz] * point_num * [ n, v, k ]
    to Tensor: 
    [n,v,k] * point_num * joint_num * [x,y,z,w]
    '''
    def get_rotation_info(self,data):
        assert len(data.shape) == 4
        assert data.shape[2] == self.point_num
        assert data.shape[0] == len(self.joints)
        assert data.shape[1] == 7
        assert data.shape[3] == 3
        rotations_info = np.zeros([3, self.point_num,len(self.joints),4])
        for time in range(self.point_num):
            for joint in range(len(self.joints)):
                for i in range(3):
                    rotations_info[i][time][joint] = [x[time][i] for x in data[joint][:4]]
        return torch.tensor(rotations_info,dtype = torch.float)

    '''
    Convert list: 
    joint_num * [x,y,z,w,px,py,pz] * point_num * [ n, v, k ]
    to Tensor : 
    [n,v,k] * point_num * joint_num * [px,py,pz]
    '''
    def get_position_info(self,data):
        assert len(data.shape) == 4
        assert data.shape[2] == self.point_num
        assert data.shape[0] == len(self.joints)
        assert data.shape[1] == 7
        assert data.shape[3] == 3
        positions_info = np.zeros([3,self.point_num,len(self.joints),3])
        for time in range(self.point_num):
            for joint in range(len(self.joints)):
                for i in range(3):
                    positions_info[i][time][joint] = [x[time][i] for x in data[joint][-3:]]
        return torch.tensor(positions_info, dtype = torch.float)
    
    def get_inputs(self, rotations_info, positions_info):
        if self.inputtype == "position":
            inputs = torch.cat((rotations_info[1],positions_info[1]), dim = 2)
        elif self.inputtype == "gradient":
            inputs = torch.cat((rotations_info[2],positions_info[2]), dim = 2)
        inputs = torch.cat((inputs[0],inputs[-1]),dim=1)
        return inputs.t()

    def get_targets(self,data):
        if self.outputtype == "deviation":
            for joint in data:
                for info in joint:
                    for time in range (1,self.point_num,1):
                        info[time][0] = info[time][0] - info[time-1][0]
        return data.view(-1)


    def __getitem__(self, index):
        data = self.cache_scheduler.load(index)["info"]
        data = self.normalize(data)
        rotations_info = self.get_rotation_info(data)
        positions_info = self.get_position_info(data)
        inputs = self.get_inputs(rotations_info,positions_info)
        # joint_num * 7 * point_num * 3
        targets = self.get_targets(data)
        return inputs, targets
        
    def get_dim(self):
        return len(self.joints) * 7 * self.point_num * 3

    def __len__(self):
        return len(self.data_files)

if __name__ == '__main__':
    motiondataset = MotionDataset("data_example\data\\","data_example\joint_info.json",16)
    x, y = motiondataset[0]
    import ipdb;ipdb.set_trace()