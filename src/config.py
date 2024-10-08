import numpy as np
import torch
import os


CHECKPOINT_DIR = "./checkpoints/my_checkpoint.pth.tar"
RESULT_DIR = "./results/"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32
NUM_WORKERS = 0
LEARNING_RATE = 0.001
WEIGHT_DECAY = 5e-3
SAVE_MODEL = True
LOAD_MODEL = True
DATASET = "shanghai"
PLOT_FREQ = 10
MAX_STEP = 15000
FEATURE_EXTRACTOR = "i3d"
FEATURE_SIZE = 2048

class Config(object) :
    def __init__(self, args) :
        self.lr = eval(args.lr)
        self.lr_str = args.lr

    def __str__(self) :
        attrs = vars(self)
        attrs_lst = sorted(attrs.keys())
        return '\n'.join("- %s: %s" % (item, attrs[item]) for item in attrs_lst if item != 'lr')    
