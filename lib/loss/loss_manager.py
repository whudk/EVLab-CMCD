##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: DonnyYou, RainbowSecret, JingyiXie, JianyuanGuo
## Microsoft Research
## yuyua@microsoft.com
## Copyright (c) 2019
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from lib.loss.loss_contrast_mem import ContrastChangeLoss


class LossManager:

    def __init__(self, configer):
        self.configer = configer

    def build_loss(self, loss_type=None, model=None):
        if loss_type != "contrast_change_loss":
            raise ValueError("This extraction supports contrast_change_loss only.")
        return ContrastChangeLoss(self.configer)
