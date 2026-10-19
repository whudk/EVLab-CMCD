from lib.models.backbones.vision_transformer.vision_transformer import vit_base, vit_small

arch_dict = {"vit_base": vit_base, "vit_small": vit_small}


class VitBackbone(object):

    def __init__(self, configer=None, masked_im_modeling=False, ret_feats=False):
        self.configer = configer
        self.masked_im_modeling = masked_im_modeling
        self.ret_feats = ret_feats

    def __call__(self, arch=None, net_params=None):
        assert arch is not None and net_params is not None
        net_params["masked_im_modeling"] = self.masked_im_modeling
        net_params["ret_feats"] = self.ret_feats
        if arch in arch_dict:
            arch_net = arch_dict[arch](**net_params)
        else:
            raise Exception("Architecture undefined!")
        return arch_net
