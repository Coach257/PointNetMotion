import argparse
import os 
import time
from addict import Dict
from torch.utils.tensorboard import SummaryWriter
import yaml

from datasets.data_loader import build_dataloader
from model.models_point_net import PointNetCls
from utils.logger import Logger
import torch.nn.functional as F

def parse_args():
    parser = argparse.ArgumentParser(description="Train motion PointNet model")
    parser.add_argument("config", help="train config file path")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--resume_from", help="the checkpoint file to resume from")
    return parser.parse_args()

class Trainer(object):
    def __init__(self,cfg):
        super().__init__()
        self.cfg = cfg
        self.logger = Logger("Trainer")
        cfg.loss.device = torch.device(cfg.device)
        os.makedirs(cfg.train.save_path,exist_ok=True)
        os.makedirs(cfg.train.log_dir,exist_ok=True)
        self.writer = SummaryWriter(log_dir=cfg.train.log_dir)

    def train(self):
        



def main():
    args = parse_args()
    cfg = Dict(yaml.safe_load(open(args.config)))
    cfg.device = args.gpu_id
    cfg.resume_from = args.resume_from
    trainer = Trainer(cfg)
    trainer.train()

if __name__ == "__main__":
    main()

