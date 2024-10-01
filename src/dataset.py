import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
from utils import *

class ShanghaiDataset(Dataset):
    def __init__(self, train=True, is_normal=True, transform=None):
        super(ShanghaiDataset, self).__init__()
        self.train = train
        self.is_normal = is_normal
        self.transform = transform
        if self.train:
            self.datapath = "./list/SH_Train_ten_crop_i3d/SH_Train_ten_crop_i3d/"
            self.list = list(open(('./list/shanghai-i3d-train-10crop.list')))
        else:
            self.datapath = "./list/SH_Test_ten_crop_i3d/SH_Test_ten_crop_i3d/"
            self.list = list(open(('./list/shanghai-i3d-test-10crop.list')))
        
        if self.train:
            if self.is_normal:
                self.list = self.list[63:]
            else:
                self.list = self.list[:63]        

        # print(self.list)
        
    def __len__(self):
        return len(self.list)
    
    def get_label(self):
        if self.is_normal:
            label = torch.tensor(0.0)
        else:
            label = torch.tensor(1.0)
        return label

    def __getitem__(self, index):
        path = self.list[index].strip('\n')
        # print(path)
        features = np.load(path, allow_pickle=True)
        features = np.array(features, dtype=np.float32)
        label = self.get_label()

        if self.transform is not None:
            features = self.transform(features)

        if not self.train:
            return features
        
        features = features.transpose(1, 0, 2)
        divided_features = [process_feat(feature, 32) for feature in features]
        divided_features = np.array(divided_features, dtype=np.float32)

        return divided_features, label
    

if __name__ == "__main__":
    shanghai_test_dataset = ShanghaiDataset(train=False, transform=None)
    shanghai_test_dataloader = DataLoader(shanghai_test_dataset, batch_size=1, shuffle=False, pin_memory=False, num_workers=0)
    c = 0

    for feature in shanghai_test_dataloader:
        print(feature.shape)
        # break
        c+=1

    print(c)

    # print (shanghai_test_dataset[15].shape)
