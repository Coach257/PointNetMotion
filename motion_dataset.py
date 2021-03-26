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
        assert inputtype in __inputtype__
        assert outputtype in __outputtype__
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
    Convert list: joint_num * [x,y,z,w] * point_num * [ [n, v], k ]
    to Tensor: 
    n : point_num * joint_num * [xn,yn,zn,wn]
    v : point_num * joint_num * [xv,yv,zv,wv]
    k : point_num * joint_num * [xk,yk,zk,wk]
    '''
    def get_local_rotation_info(self,data):
        data = np.array(data)
        assert data.shape[2] == self.point_num
        assert data.shape[0] == len(self.joints)
        assert data.shape[1] == 4
        local_rotations_v = np.zeros([self.point_num,len(self.joints),4])
        local_rotations_n = np.zeros([self.point_num,len(self.joints),4])
        local_rotations_k = np.zeros([self.point_num,len(self.joints),4])
        for time in range(self.point_num):
            for joint in range(len(self.joints)):
                local_rotations_v[time][joint] = [x[time][0][1] for x in data[joint]]
                local_rotations_n[time][joint] = [x[time][0][0] for x in data[joint]]
                local_rotations_k[time][joint] = [x[time][1] for x in data[joint]]
        return torch.tensor(local_rotations_n,dtype = torch.double), torch.tensor(local_rotations_v,dtype = torch.double), torch.tensor(local_rotations_k,dtype = torch.double)

    '''
    Convert list: [x,y,z] * point_num * [ [n,v], k ]
    to Tensor : 
    n : point_num * [xn,yn,zn]
    v : point_num * [xv,yv,zv]
    k : point_num * [xk,yk,zk]
    '''
    def get_root_position_info(self,data):
        data = np.array(data)
        assert data.shape[0] == 3
        assert data.shape[1] == self.point_num
        root_positions_n = np.zeros([self.point_num,3])
        root_positions_v = np.zeros([self.point_num,3])
        root_positions_k = np.zeros([self.point_num,3])
        for time in range(self.point_num):
            root_positions_n[time] = [x[time][0][0] for x in data]
            root_positions_v[time] = [x[time][0][1] for x in data]
            root_positions_k[time] = [x[time][1] for x in data]
        return torch.tensor(root_positions_n, dtype = torch.double),torch.tensor(root_positions_v, dtype = torch.double),torch.tensor(root_positions_k, dtype = torch.double)
    
    def get_inputs(self, local_rotations_v,local_rotations_k, global_pos):
        if self.inputtype == "position":
            inputs = torch.cat((local_rotations_v,global_pos),dim=2)
            inputs = torch.cat((inputs[0],inputs[-1]),dim=1)
            return inputs.t()
        elif self.inputtype == "gradient":
            pass

    def __getitem__(self, index):
        data = self.cache_scheduler.load(index)
        local_rotations_n, local_rotations_v, local_rotations_k = self.get_local_rotation_info(data["rotations"])
        root_positions_n, root_positions_v,root_positions_k = self.get_root_position_info(data["root_positions"])
        global_pos = torch.tensor(self.sct.forward_kinematics(local_rotations_v, root_positions_v, rot_type="local"),dtype=torch.double)
        inputs = self.get_inputs(local_rotations_v,local_rotations_k, global_pos)
        return inputs
        
    def __len__(self):
        return len(self.data_files)

if __name__ == '__main__':
    motiondataset = MotionDataset("data_example\data\\","data_example\joint_info.json",16)
    x = motiondataset[0]
    import ipdb;ipdb.set_trace()