#! /usr/bin/env python
# coding=utf-8
# ================================================================
#   Copyright (C) 2023 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : resnet.py
#   Author      : YunYang1994
#   Created date: 2023-08-03 18:23:08
#   Description :
#
# ================================================================


import torchvision
import torch.nn.functional as F
from mmengine.model import BaseModel
from .builder import MODELS

@MODELS.register_module()
class ResNet18(BaseModel):
    def __init__(self):
        super().__init__()
        self.resnet = torchvision.models.resnet18()

    def forward(self, imgs, labels, mode):
        x = self.resnet(imgs)
        if mode == 'loss':
            return {'loss': F.cross_entropy(x, labels)}
        elif mode == 'predict':
            return x, labels