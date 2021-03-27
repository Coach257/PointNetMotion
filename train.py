import argparse
import os 
import time
import torch
from addict import Dict
from torch.utils.tensorboard import SummaryWriter
import yaml

from datasets.data_loader import build_dataloader
from model.models_point_net import PointNetCls
from utils.logger import Logger
from model.models_point_net import trans_regularizer
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
        cfg = self.cfg
        device = torch.device(cfg.device)
        data_loader = build_dataloader(cfg.data.data_path, cfg.data.joint_file, cfg.train.batch_size,cfg.train.point_num,cfg.train.inputtype,cfg.train.outputtype)
        dataset = data_loader.dataset
        cfg.model.n_cls = dataset.get_dim()

        model = PointNetCls(cfg.model)
        model.to(device=device)

        if cfg.train.optimizer.type == "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.optimizer.lr,
                betas=(cfg.train.optimizer.get("beta1", 0.9), cfg.train.optimizer.get("beta2", 0.999)), 
                weight_decay=cfg.train.optimizer.get("weight_decay", 0),
                amsgrad=cfg.train.optimizer.get("amsgrad", False))
        elif cfg.train.optimizer.type == "AdamW":
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.optimizer.lr,
                betas=(cfg.train.optimizer.get("beta1", 0.9), cfg.train.optimizer.get("beta2", 0.999)), 
                weight_decay=cfg.train.optimizer.get("weight_decay", 0),
                amsgrad=cfg.train.optimizer.get("amsgrad", False))
        else:
            self.logger.logging("The optimizer type {} is not supported".format(cfg.train.optimizer.type))

        losses = 0.
        step = 0
        best_metric = 1e7
        start_epoch = 0

        if cfg.resume_from is not None:
            state = torch.load(cfg.resume_from)
            model.load_state_dict(state['net'])
            optimizer.load_state_dict(state['optimizer'])
            start_epoch = state['epoch'] + 1
            best_metric = state['best_metric']
            step = state['step']

        for e in range(start_epoch, cfg.train.epoch):
            model.train()
            self.logger.logging("Start Epoch %d " %e)
            metric = 0.
            for input,target in data_loader:
                input = input.to(device)
                target = target.to(device)
                pred, trans_mtx = model(input)


                loss = F.mse_loss(pred,target) + trans_regularizer(trans_mtx,device)*0.001
                self.writer.add_scalar("loss", loss.item(), step)
                losses += loss.item()
                step += 1
                metric += loss.item()
                if step % cfg.train.disp_every == 0:
                    self.logger.logging("Step {}, loss is {}".format(step, losses / cfg.train.disp_every))
                    losses = 0.
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            metric /= len(data_loader)
            self.writer.add_scalar("metric", metric, e)
            self.logger.logging("Epoch {}, the metric is {}".format(e, metric))
            if metric < best_metric:
                self.logger.logging("Epoch {}, the best metric {} --> {}".format(e, best_metric, metric))
                best_metric = metric
                state = {}
                state["config"] = cfg
                state["net"] = model.state_dict()
                torch.save(state, os.path.join(cfg.train.save_path, "best.m"))
            
            if e >= cfg.train.epoch - 10:
                state = {}
                state["config"] = cfg
                state["net"] = model.state_dict()
                torch.save(state, os.path.join(cfg.train.save_path, "model_{}.m".format(e))) 
            
            state = {}
            state['config'] = cfg
            state['net'] = model.state_dict()
            state['optimizer'] = optimizer.state_dict()
            state['best_metric'] = best_metric
            state['epoch'] = e
            state['step'] = step
            torch.save(state, os.path.join(cfg.train.save_path, "checkpoint.m"))

def main():
    args = parse_args()
    cfg = Dict(yaml.safe_load(open(args.config)))
    cfg.device = args.gpu_id
    cfg.resume_from = args.resume_from
    trainer = Trainer(cfg)
    trainer.train()

if __name__ == "__main__":
    main()

