#! /usr/bin/env python
# coding=utf-8
# ================================================================
#   Copyright (C) 2023 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : cifar10.py
#   Author      : YunYang1994
#   Created date: 2023-08-03 18:03:18
#   Description :
#
# ================================================================


import torchvision
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from .builder import DATASETS

@DATASETS.register_module()
class CifarDataset(Dataset):
    def __init__(self, root_dir, norm_cfg, training=True, download=True):
        super(CifarDataset, self).__init__()
        if training:
            transform=transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(**norm_cfg)])
        else:
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(**norm_cfg)])
        self.dataset = torchvision.datasets.CIFAR10(
                            root_dir,
                            train=training,
                            download=download,
                            transform=transform
                        )
    
    def __getitem__(self, index):
        return self.dataset[index]
    
    def __len__(self):
        return len(self.dataset)
