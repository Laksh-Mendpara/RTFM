import numpy as np
import os

class Config(object) :
    def __init__(self, args) :
        self.lr = eval(args.lr)
        self.lr_str = args.lr

    def __str__(self) :
        attrs = vars(self)
        attrs_lst = sorted(attrs.keys())
        return '\n'.join("- %s: %s" % (item, attrs[item]) for item in attrs_lst if item != 'lr')    