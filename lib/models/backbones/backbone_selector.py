##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Donny You, RainbowSecret
## Microsoft Research
## yuyua@microsoft.com
## Copyright (c) 2019
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from lib.models.backbones.vision_transformer.vit_backbone import VitBackbone


class BackboneSelector:

    def __init__(self, configer):
        self.configer = configer

    def get_backbone(self, backbone=None, **params):
        if backbone not in ("vit_base", "vit_small"):
            raise ValueError("Supported backbones: vit_small, vit_base.")
        return VitBackbone(self.configer)(**params)
