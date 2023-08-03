#! /usr/bin/env python
# coding=utf-8
# ================================================================
#   Copyright (C) 2023 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : metric.py
#   Author      : YunYang1994
#   Created date: 2023-08-03 19:53:12
#   Description :
#
# ================================================================

from mmengine.evaluator import BaseMetric
from .builder import MODELS

@MODELS.register_module()
class Accuracy(BaseMetric):
    def process(self, data_batch, data_samples):
        score, gt = data_samples
        self.results.append({
            'batch_size': len(gt),
            'correct': (score.argmax(dim=1) == gt).sum().cpu(),
        })

    def compute_metrics(self, results):
        total_correct = sum(item['correct'] for item in results)
        total_size = sum(item['batch_size'] for item in results)
        return dict(accuracy=100 * total_correct / total_size)