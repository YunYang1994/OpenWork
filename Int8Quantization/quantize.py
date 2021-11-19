#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2020 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : quantize.py
#   Author      : YunYang1994
#   Created date: 2020-04-16 12:00:58
#   Description :
#
#================================================================

import copy
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from scipy import stats
from train import VGG16, CifarDataset

"""
http://on-demand.gputechconf.com/gtc/2017/presentation/s7310-8-bit-inference-with-tensorrt.pdf
https://github.com/Ewenwan/MVision/tree/master/CNN/Deep_Compression/quantization
"""

class QuantizeLayer:
    def __init__(self, name, layer, grids=2048):
        self.name     = name
        self.layer    = layer
        self.channels = layer.out_channels
        self.weight_scales = np.zeros(self.channels)    # each channels has one scale

        self.blob_scale = 1.
        self.blob_max   = 0.
        self.grids      = grids
        self.blob_count = np.zeros(self.grids)

    def hook(self, modules, input):
        """
        VGG16 模型每次 forward 时，都会调用 QuantizeLayer.hook 函数，更新数据流的直方图分布
        """
        self.blob = input[0].cpu().detach().numpy().flatten()
        max_val = np.max(self.blob)
        min_val = np.min(self.blob)
        self.blob_max = max(self.blob_max, max(abs(max_val), abs(min_val)))

        # 将数据的绝对值范围 (0, blob_max) 划分为 2048 个区间，然后计算每个区间内的数据的总数, 即一个直方图分布
        count, _ = np.histogram(self.blob, bins=self.grids, range=(0, self.blob_max))
        self.blob_count = self.blob_count + count

        threshold_bin = self.quantize_blob()
        threshold_val = (threshold_bin + 0.5) * (self.blob_max / 2048)
        self.blob_scale = 127 / threshold_val
        print("%-10s threshold_bin: %-3d  blob_max : %-10f  blob_scale: %-10f threshold_val: %-10f"
                %(self.name+"_param0", threshold_bin, self.blob_max, self.blob_scale, threshold_val))

    def quantize_weight(self):
        """
        对该层的卷积核权重进行量化, 计算出 scale
        """
        weights = self.layer.weight.cpu().detach().numpy()
        group_weights = np.array_split(weights, self.channels)

        print("-"*50 + " %-5s " %self.name + "-"*50)
        for i, group_weight in enumerate(group_weights):
            max_val = np.max(group_weight)
            min_val = np.min(group_weight)

            thresh  = max(abs(max_val), abs(min_val))
            if thresh < 0.0001:
                self.weight_scales[i] = 0.
            else:
                self.weight_scales[i] = 127 / thresh # int8: -127 ~ 127
            print("%-10s  group : %-3d min_val : %-10f max_val : %-10f thresh : %-10f scale : %-10f"
                    % (self.name+"_param0", i, min_val, max_val, thresh, self.weight_scales[i]))

    def quantize_blob(self):
        """
        对该层的输入数据流进行量化, 计算出 scale
        """
        target_bin=128
        distribution = self.blob_count[1:] # 第一刻度的量化不在考虑范围内，因为它映射到 int8 为0
        length = distribution.size
        threshold_sum = sum(distribution[target_bin:])
        kl_divergence = np.zeros(length - target_bin)

        for threshold in range(target_bin, length):
            sliced_nd_hist = copy.deepcopy(distribution[:threshold])

            p = sliced_nd_hist.copy()
            p[threshold - 1] += threshold_sum # boundary sum
            threshold_sum = threshold_sum - distribution[threshold]

            is_nonzeros = (p != 0).astype(np.int64)
            quantized_bins = np.zeros(target_bin, dtype=np.int64)
            num_merged_bins = sliced_nd_hist.size // target_bin

            for j in range(target_bin):
                start = j * num_merged_bins
                stop = start + num_merged_bins
                quantized_bins[j] = sliced_nd_hist[start:stop].sum()
            quantized_bins[-1] += sliced_nd_hist[target_bin * num_merged_bins:].sum()

            q = np.zeros(sliced_nd_hist.size, dtype=np.float64)
            for j in range(target_bin):
                start = j * num_merged_bins
                if j == target_bin - 1:
                    stop = -1
                else:
                    stop = start + num_merged_bins
                norm = is_nonzeros[start:stop].sum()
                if norm != 0:
                    q[start:stop] = float(quantized_bins[j]) / float(norm)
            q[p == 0] = 0
            p[p == 0] = 0.0001
            q[q == 0] = 0.0001
            kl_divergence[threshold - target_bin] = stats.entropy(p, q)

        min_kl_divergence = np.argmin(kl_divergence)
        threshold_bin = min_kl_divergence + target_bin
        return threshold_bin


    def show_results(self):
        hist_y = self.blob_count / np.sum(self.blob_count)
        fig, ax1 = plt.subplots()
        ax1.semilogy(range(self.grids), hist_y)
        plt.xlabel("grid")
        plt.ylabel("normalized count of number")
        plt.show()






model_path = "/home/yyang/VGG16-testAcc=0.7915.pth"
Qlayers    = []
model      = VGG16(num_class=10)
state_dict = torch.load(model_path, map_location=torch.device('cpu'))
state_dict = {k:v for k,v in state_dict.items() if k in model.state_dict()}
model.load_state_dict(state_dict)
model.eval()
model.cuda()

for name, layer in model.named_modules():
    if isinstance(layer, nn.Conv2d):
        Qlayer = QuantizeLayer(name, layer)
        Qlayer.quantize_weight()
        # Qlayer.quantize_blob()

        # register hook to collect conv layer's input
        layer.register_forward_pre_hook(Qlayer.hook)
        Qlayers.append(Qlayer)

dataset = CifarDataset("/data0/yyang/cifar10/quantize")
dloader = iter(torch.utils.data.DataLoader(dataset, 1, shuffle=False))

for sample in dloader:
    image = sample['image']
    model(image.cuda())
    # break

# Qlayers[3].show_results()
