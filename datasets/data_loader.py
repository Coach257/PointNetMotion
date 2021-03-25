import numpy as np
import torch
from torch.utils.data import DataLoader, BatchSampler, SequentialSampler
from .motion_dataset import MotionDataset

def build_dataloader(data_path, joint_file, batch_size, point_num, inputtype, outputtype):
    dataset = MotionDataset(data_path, joint_file, point_num, inputtype, outputtype)
    return DataLoader(dataset, sampler=BatchSampler(sampler=SequentialSampler(dataset),batch_size = batch_size, drop_last = True))


if __name__ == "__main__":
    data_loader = build_dataloader("/home/data/Motion3D/motionjson/", "joint_info.json", 32,16)
    input,target,mask = next(iter(data_loader))