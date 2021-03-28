import torch
import json
import numpy as np
import os
from utils.data_util import get_all_files
import random
import argparse
from addict import Dict
import yaml
from model.models_point_net import PointNetCls

random.seed(a=None, version=2)
def parse_args():
    parser = argparse.ArgumentParser(description="Predict motion with PointNetMotion model")
    parser.add_argument("config", help="Prediction config file path")
    return parser.parse_args()

class Predictor(object):
    def __init__(self,cfg):
        self.cfg = cfg
        state_dict = torch.load(self.cfg.model_path,map_location=torch.device('cpu'))
        self.model_cfg = state_dict["config"]
        self.model = PointNetCls(self.model_cfg.model)
        self.model.load_state_dict(state_dict['net'])
        self.model.eval()
        self.device = torch.device(self.cfg.device)
        self.model.to(self.device)
        joint_info = json.load(open(self.cfg.data.joint_file))
        self.joints = joint_info['joints']
        self.point_num = cfg.point_num
        self.inputtype = cfg.inputtype
        self.max_len = cfg.max_len

    def unnormalize(self,data):
        data[:][:][:] = data[:][:][:]*torch.tensor([self.max_len,1,1])
        return data
    def normalize(self,data):
        data = torch.tensor(data , dtype = torch.float)
        data[:][:][:] = data[:][:][:]/torch.tensor([self.max_len,1,1])
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
    
    def predict(self,data):
        data = self.normalize(data)
        rotations_info = self.get_rotation_info(data)
        positions_info = self.get_position_info(data)
        inputs = self.get_inputs(rotations_info,positions_info)
        inputs = inputs.view(1,14,len(self.joints))
        pred,trans_mtx = self.model(inputs)
        pred = pred.view(len(self.joints),7,self.point_num,3)
        pred = self.unnormalize(pred)
        return pred

def main():
    args = parse_args()
    cfg = Dict(yaml.safe_load(open(args.config)))
    cfg.device = 'cpu'
    data_files = get_all_files(cfg.data.data_path)
    testcase_path = data_files[random.randint(0,len(data_files))]
    save_path = os.path.join(args.config.replace("predict_config.yml",""), testcase_path)
    print("Testcase : {}".format(testcase_path))
    print("Save dir : {}".format(save_path))
    
    data = json.load(open(os.path.join(cfg.data.data_path,testcase_path),"r"))
    cfg.max_len = data["info"][0][0][-1][0]
    predictor = Predictor(cfg)
    pred = predictor.predict(data["info"]).detach().numpy().tolist()
    os.makedirs(save_path)
    json.dump(data,open(os.path.join(save_path,"origin.json"),"w"))
    json.dump({"name":data["name"],"info":pred},open(os.path.join(save_path,"out.json"),"w"))

if __name__ == "__main__":
    main()   