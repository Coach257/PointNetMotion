import torch
import json
import numpy as np
import os
from utils.data_util import get_all_files
import random
import argparse
from addict import Dict
import yaml
from model.models_point_net import PointNetCls, trans_regularizer
import torch.nn.functional as F
from data_process.Rehermite import ReHermite

def parse_args():
    parser = argparse.ArgumentParser(description="Predict motion with PointNetMotion model")
    parser.add_argument("config", help="Prediction config file path")
    return parser.parse_args()

class Evaluator(object):
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
        
        self.data_path = cfg.data.data_path
        self.data_files = get_all_files(cfg.data.data_path)
        sample = json.load(open(os.path.join(self.data_path,self.data_files[0])))
        self.name = sample["name"]
        self.max_len = sample["info"][0][0][-1][0]
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
            inputs = torch.cat((rotations_info[1], rotations_info[2],positions_info[1],positions_info[2]), dim = 2)
        inputs = torch.cat((inputs[0],inputs[-1]),dim=1)
        return inputs.t()


    def get_pred(self,data, pred):
        res = torch.zeros(len(self.joints),7,self.point_num,3)
        for i in range(len(self.joints)):
            for j in range(7):
                res[i][j][0] = data[i][j][0]
                res[i][j][self.point_num-1] = data[i][j][self.point_num-1]
                for k in range(self.point_num-2):
                    res[i][j][k+1] = pred[i][j][k]
        return res

    def evaluate(self):
        losses = 0.
        min_loss = 5.0
        best_name = ""
        best_out = torch.zeros(len(self.joints),7,self.point_num,3)
        for data_file in self.data_files:
            print("Evaluate {}".format(data_file))
            data = json.load(open(os.path.join(self.data_path,data_file)))["info"]
            data = self.normalize(data)
            rotations_info = self.get_rotation_info(data)
            positions_info = self.get_position_info(data)
            inputs = self.get_inputs(rotations_info,positions_info)
            if self.inputtype == "position":
                inputs = inputs.view(1,14,len(self.joints))
            elif self.inputtype == "gradient":
                inputs = inputs.view(1,28,len(self.joints))

            pred, trans_mtx = self.model(inputs)
            pred = pred.view(len(self.joints),7,self.point_num-2,3)
            pred = self.get_pred(data,pred)
            loss = F.mse_loss(pred, data) + trans_regularizer(trans_mtx, "cpu")*0.001
            losses += loss
            if loss < min_loss:
                min_loss = loss
                best_name = data_file
                best_out = pred
        best_out = best_out.view(len(self.joints),7,self.point_num,3)
        best_out = self.unnormalize(best_out)
        losses = losses / len(self.data_files)
        return losses, min_loss, best_name, {"name":self.name,"info":best_out.detach().numpy().tolist()}

def main():
    args = parse_args()
    cfg = Dict(yaml.safe_load(open(args.config)))
    cfg.device = 'cpu'
    evaluator = Evaluator(cfg)
    loss, best_loss,best_name, best_outcome = evaluator.evaluate()
    print("The average loss is {}".format(loss))
    print("The best loss is {},named {}".format(best_loss,best_name))
    ReHermite(best_outcome,os.path.join(args.config.replace(args.config.split("/")[-1],""),"best_outcome.json"))
    

if __name__ == "__main__":
    main()   