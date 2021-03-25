import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['PointNetfeat', 'PointNetCls', 'trans_regularizer']


class STNkd(nn.Module):
    def __init__(self, k=14, mode='init_identity'):
        super().__init__()
        self.k = k
        self.mode = mode

        self.mlp1 = nn.Sequential(
            nn.Conv1d(k, 64, 1, bias=False), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 1, bias=False), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 1024, 1, bias=False), nn.BatchNorm1d(1024), nn.ReLU())
        self.mlp2 = nn.Sequential(
            nn.Linear(1024, 512, bias=False), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, 256, bias=False), nn.BatchNorm1d(256), nn.ReLU())
        self.fc_trans = nn.Linear(256, k*k)

        if mode == 'init_identity':
            self.fc_trans.bias.data.fill_(0)
            self.fc_trans.weight.data.fill_(0)
            for cnt in range(k):
                self.fc_trans.bias.data[cnt*k+cnt] = 1

    def forward(self, x):
        n_pts = x.shape[2]
        batchsize = x.shape[0]
        x = self.mlp1(x)
        x = F.max_pool1d(x, n_pts).squeeze(2)
        x = self.mlp2(x)
        x = self.fc_trans(x)
        if self.mode == 'add_identity':
            iden = torch.eye(self.k, dtype=torch.float32).cuda().view(
                1, -1).repeat(batchsize, 1)
            x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetfeat(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.input_trans = STNkd(k=3, mode=cfg.mode)
        self.mlp1 = nn.Sequential(
            nn.Conv1d(3, 64, 1, bias=False), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 64, 1, bias=False), nn.BatchNorm1d(64), nn.ReLU())
        self.feat_trans = STNkd(k=64, mode=cfg.mode)
        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 64, 1, bias=False), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 1, bias=False), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 1024, 1, bias=False), nn.BatchNorm1d(1024), nn.ReLU())

    def forward(self, x):
        n_pts = x.shape[2]
        trans_input_mtx = self.input_trans(x)
        x = torch.bmm(trans_input_mtx, x)
        x = self.mlp1(x)
        trans_feat_mtx = self.feat_trans(x)
        x = torch.bmm(trans_feat_mtx, x)
        point_feat = self.mlp2(x)
        return point_feat, trans_feat_mtx 


class PointNetCls(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.feat = PointNetfeat(cfg)
        self.mlp = nn.Sequential(
            nn.Linear(1024, 512, bias=False), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, 256, bias=False), nn.BatchNorm1d(256), nn.ReLU())
        self.dropout = nn.Dropout(p=0.3)
        self.fc_cls = nn.Linear(256, cfg.n_cls)
        self.fc_reg = nn.Linear(256, cfg.n_cls)

    def forward(self, x):
        n_pts = x.shape[2]
        x, trans_mtx = self.feat(x)
        x = F.max_pool1d(x, n_pts).squeeze(2)
        x = self.mlp(x)
        x = self.dropout(x)
        cls_pred = self.fc_cls(x)
        reg_pred = self.fc_reg(x)
        return cls_pred, reg_pred, trans_mtx


def trans_regularizer(trans):
    trans_square = torch.bmm(trans, trans.transpose(1, 2))
    d = trans_square.shape[1]
    identity = torch.eye(d).cuda().expand_as(trans_square)
    loss = F.mse_loss(trans_square, identity)
    return loss


'''
    example
'''
if __name__ == '__main__':
    ## 关于input： 位置还是nvk
    ## 关于首位拼接 cat还是直接拼接
    inputs = torch.rand([32, 3, 34]).cuda()
    cls_labels = torch.randint(10, [32]).cuda()
    reg_labels = torch.randn([32]).cuda()
    cls_pred, reg_pred, trans_mtx = net(inputs)

    loss = F.cross_entropy(cls_pred, cls_labels) + \
        F.mse_loss(reg_pred, reg_labels) + \
        trans_regularizer(trans_mtx) * 0.001
        