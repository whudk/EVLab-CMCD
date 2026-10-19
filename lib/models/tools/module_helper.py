#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You (youansheng@gmail.com)

import torch.nn as nn


class ModuleHelper:

    @staticmethod
    def BatchNorm2d(bn_type="torchbn", ret_cls=False):
        if bn_type == "torchbn":
            return nn.BatchNorm2d
        if bn_type == "torchsyncbn":
            return nn.SyncBatchNorm
        raise ValueError(f"Unsupported normalization: {bn_type}")

    @staticmethod
    def BNReLU(num_features, bn_type="torchbn", **kwargs):
        return nn.Sequential(
            ModuleHelper.BatchNorm2d(bn_type)(num_features, **kwargs), nn.ReLU()
        )
