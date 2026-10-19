##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Microsoft Research
## Author: RainbowSecret, LangHuang, JingyiXie, JianyuanGuo
## Copyright (c) 2019
## yuyua@microsoft.com
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from lib.models.backbones.vision_transformer.vit_backbone import VitBackbone
from lib.models.decoder.cd_head import CDHead_shp_to_img


class ModelManager:

    def __init__(self, configer):
        self.configer = configer

    def build_encoder(self, encoder_name=None, net_params=None, **kwargs):
        return VitBackbone(self.configer, **kwargs)(
            arch=encoder_name, net_params=net_params
        )

    def build_decoder(self, name=None, **kwargs):
        cfg = self.configer.get("network", "decoder")
        if (name or cfg["name"]) != "head_shpcd":
            raise ValueError("This extraction supports head_shpcd only.")
        return CDHead_shp_to_img(configer=self.configer, **cfg["params"])
