import torch
import torch.nn as nn
from torch.nn import Sigmoid
import torch.nn.functional as F
import numpy as np
torch.set_default_tensor_type('torch.FloatTensor')
from torch.nn import L1Loss
from torch.nn import MSELoss
import config


def sparsity(arr, lambda2):
    return lambda2*torch.mean(torch.norm(arr, dim=0))

def smooth(arr, lambda1):
    arr2 = torch.zeros_like(arr)
    arr2[:-1] = arr[1:]
    arr2[-1] = arr[-1]
    return lambda1*torch.sum((arr2-arr)**2)

def l1_penalty(var):
    return torch.mean(torch.norm(var, dim=0))

class SigmoidMAELoss(nn.Module):
    def __init__(self):
        super(SigmoidMAELoss, self).__init__()
        self.sig = Sigmoid()
        self.mse = MSELoss()

    def forward(self, tar, pred):
        return self.mse(tar, pred)
    
class SigmoidCrossEntrpyLoss(nn.Module):
    def __init__(self):
        super(SigmoidCrossEntrpyLoss, self).__init__()

    def forward(self, x, target):
        denominator = 1 + torch.exp(-torch.abs(x))
        return torch.abs(torch.mean(- x*target + torch.clamp(x, min=0) + torch.log(denominator)))

class RTFM_loss(nn.Module):
    def __init__(self, alpha, margin):
        super(RTFM_loss, self).__init__()
        self.alpha = alpha
        self.margin = margin
        self.sigmoid = nn.Sigmoid()
        self.mae = SigmoidMAELoss()
        self.criterion = nn.BCELoss()

    def forward(self, nscore, ascore, nlabel, alabel, nfeat, afeat):
        label = torch.cat((nlabel, alabel), dim=0).to(config.DEVICE)
        score = torch.cat((nscore, ascore), dim=0)
        score = score.squeeze()

        loss_cls = self.criterion(score, label)
        loss_abn = torch.abs(self.margin - torch.norm(torch.mean(afeat, dim=1), p=2, dim=1))
        loss_nor = torch.norm(torch.mean(nfeat, dim=1), p=2, dim=1)

        loss_rtfm = torch.mean((loss_abn+loss_nor)**2)
        return loss_cls+self.alpha*loss_rtfm
    
def train(nloader, aloader, model, batch_size, optimizer):
    with torch.set_grad_enabled(True):
        model.train()

        ninput, nlabel = next(nloader)
        ainput, alabel = next(aloader)
        nlabel = nlabel[0:config.BATCH_SIZE]
        alabel = alabel[0:config.BATCH_SIZE]
        input = torch.cat((ninput, ainput), dim=0).to(config.DEVICE)
        score_abnormal, score_normal, \
        feat_select_abn, feat_sel_norm, feat_select_abn, \
            feat_select_abn, x, feat_select_abn, feat_select_abn, feat_mag = model(input)
        scores = x.view(config.BATCH_SIZE*32*2, -1)
        scores = scores.squeeze()
        abn_score = scores[config.BATCH_SIZE*32:]

        loss_criterion = RTFM_loss(0.0001, 100)
        loss_sparse = sparsity(abn_score, 8e-3)
        loss_smooth = smooth(abn_score, 8e-4)
        cost = loss_criterion(score_normal, score_abnormal, nlabel, alabel, feat_sel_norm, feat_select_abn)+loss_sparse+loss_smooth

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()


if __name__ == "__main__":
    from model import *
    from dataset import ShanghaiDataset
    from torch.utils.data import DataLoader
    import torch.optim as optim
    ndata = ShanghaiDataset(train=True, is_normal=True)
    adata = ShanghaiDataset(train=True, is_normal=False)
    nloader = DataLoader(ndata, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, 
                         pin_memory=False, drop_last=True)
    
    aloader = DataLoader(adata, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, 
                         pin_memory=False, drop_last=True)

    model = Model(config.FEATURE_SIZE, config.BATCH_SIZE).to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr = config.LEARNING_RATE, weight_decay=0.005)
    train(iter(nloader), iter(aloader), model, config.BATCH_SIZE, optimizer)
    