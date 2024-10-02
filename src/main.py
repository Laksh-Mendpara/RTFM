import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from utils import *
from model import *
from dataset import *
import config
from test import *
from train import *
import numpy as np


def main(nloader, aloader, testloader, model, optimizer, test_acc):
    
    auc = test(testloader, model)
    test_acc.append(auc)

    for step in tqdm(
        range(1, config.MAX_STEP),total=config.MAX_STEP, dynamic_ncols=True
    ):
        if (step-1)%len(nloader) == 0:
            nloaderiter = iter(nloader)

        if (step-1)%len(aloader) == 0:
            aloaderiter = iter(aloader)

        train(nloaderiter, aloaderiter, model, config.BATCH_SIZE, optimizer)

        if step%5==0:
            auc = test(testloader, model)
            test_acc.append(auc)
            print(auc)

        if step%10==0 and config.SAVE_MODEL:
            save_checkpoint(model, optimizer, config.CHECKPOINT_DIR)


if __name__ == "__main__":
    nDataset = ShanghaiDataset(train=True, is_normal=True)
    aDataset = ShanghaiDataset(train=True, is_normal=False)
    testDataset = ShanghaiDataset(train=False)
    
    nloader = DataLoader(nDataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, 
                         pin_memory=False, drop_last=True)
    aloader = DataLoader(aDataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, 
                         pin_memory=False, drop_last=True)
    testloader = DataLoader(testDataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)

    model = Model(config.FEATURE_SIZE, config.BATCH_SIZE).to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_DIR, model, optimizer, config.LEARNING_RATE)

    test_acc = []
    main(nloader, aloader, testloader, model, optimizer, test_acc)
