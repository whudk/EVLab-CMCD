##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Donny You, RainbowSecret
## Microsoft Research
## yuyua@microsoft.com
## Copyright (c) 2019
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F


class FSOhemCELoss(nn.Module):

    def __init__(self, configer, reduction="mean"):
        super(FSOhemCELoss, self).__init__()
        self.configer = configer
        self.thresh = self.configer.get("loss", "params")["ohem_thresh"]
        self.min_kept = max(1, self.configer.get("loss", "params")["ohem_minkeep"])
        weight = None
        if self.configer.exists("loss", "params") and "ce_weight" in self.configer.get(
            "loss", "params"
        ):
            weight = self.configer.get("loss", "params")["ce_weight"]
            weight = torch.FloatTensor(weight).cuda()
        self.reduction = reduction
        if self.configer.exists(
            "loss", "params"
        ) and "ce_reduction" in self.configer.get("loss", "params"):
            self.reduction = self.configer.get("loss", "params")["ce_reduction"]
        ignore_index = -1
        if self.configer.exists(
            "loss", "params"
        ) and "ce_ignore_index" in self.configer.get("loss", "params"):
            ignore_index = self.configer.get("loss", "params")["ce_ignore_index"]
        self.ignore_label = ignore_index
        self.ce_loss = nn.CrossEntropyLoss(
            weight=weight, ignore_index=ignore_index, reduction="none"
        )
        self.num_classes = self.configer.get("data", "num_classes")

    def forward(self, predict, target, **kwargs):
        """
        Args:
            predict:(n, c, h, w)
            target:(n, h, w)
            weight (Tensor, optional): a manual rescaling weight given to each class.
                                       If given, has to be a Tensor of size "nclasses"
        """
        b, c, h, w = predict.shape
        predict = F.interpolate(
            predict,
            size=(target.shape[1], target.shape[2]),
            mode="bilinear",
            align_corners=True,
        )
        prob_out = F.softmax(predict, dim=1)
        tmp_target = target.clone()
        tmp_target[tmp_target == self.ignore_label] = 0
        prob = prob_out.gather(1, tmp_target.unsqueeze(1))
        mask = target.contiguous().view(-1) != self.ignore_label
        mask[0] = 1
        sort_prob, sort_indices = prob.contiguous().view(-1)[mask].contiguous().sort()
        min_threshold = sort_prob[min(self.min_kept, sort_prob.numel() - 1)]
        threshold = max(min_threshold, self.thresh)
        loss_matirx = self.ce_loss(predict, target).contiguous().view(-1)
        sort_loss_matirx = loss_matirx[mask][sort_indices]
        min_sort_prob = torch.min(sort_prob)
        if min_sort_prob >= threshold:
            return predict.sum() * 0.0
        select_loss_matrix = sort_loss_matirx[sort_prob < threshold]
        if self.reduction == "sum":
            return select_loss_matrix.sum()
        elif self.reduction == "mean":
            return select_loss_matrix.mean()
        else:
            raise NotImplementedError("Reduction Error!")
