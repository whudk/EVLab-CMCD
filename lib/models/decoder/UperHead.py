import torch
import torch.nn as nn
from lib.models.tools.module_helper import ModuleHelper
import math
from timm.models.layers import trunc_normal_
import torch.distributed as dist


class UperHead(nn.Module):

    def __init__(
        self,
        num_class=150,
        fc_dim=4096,
        use_softmax=False,
        pool_scales=(1, 2, 3, 6),
        fpn_inplanes=(96, 192, 384, 786),
        fpn_dim=256,
        feature_normalization="shared_chunk_norm1",
        **kwargs
    ):
        super(UperHead, self).__init__()
        bn_type = "torchsyncbn" if dist.is_initialized() else "torchbn"
        self.use_softmax = use_softmax
        if feature_normalization not in ("global", "shared_chunk_norm1"):
            raise ValueError(f"Unsupported feature normalization: {feature_normalization}")
        self.feature_normalization = feature_normalization
        self.ppm_pooling = []
        self.ppm_conv = []
        for scale in pool_scales:
            self.ppm_pooling.append(nn.AdaptiveAvgPool2d(scale))
            self.ppm_conv.append(
                nn.Sequential(
                    nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                    ModuleHelper.BNReLU(512, bn_type=bn_type),
                )
            )
        self.ppm_pooling = nn.ModuleList(self.ppm_pooling)
        self.ppm_conv = nn.ModuleList(self.ppm_conv)
        self.ppm_last_conv = nn.Sequential(
            nn.Conv2d(
                fc_dim + len(pool_scales) * 512,
                fpn_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            ModuleHelper.BNReLU(fpn_dim, bn_type=bn_type),
        )
        self.fpn_in = []
        for fpn_inplane in fpn_inplanes[:-1]:
            self.fpn_in.append(
                nn.Sequential(
                    nn.Conv2d(fpn_inplane, fpn_dim, kernel_size=1, bias=False),
                    ModuleHelper.BNReLU(fpn_dim, bn_type=bn_type),
                )
            )
        self.fpn_in = nn.ModuleList(self.fpn_in)
        self.fpn_out = []
        for i in range(len(fpn_inplanes) - 1):
            self.fpn_out.append(
                nn.Sequential(
                    nn.Conv2d(
                        fpn_dim, fpn_dim, kernel_size=3, stride=1, padding=1, bias=False
                    ),
                    ModuleHelper.BNReLU(fpn_dim, bn_type=bn_type),
                )
            )
        self.fpn_out = nn.ModuleList(self.fpn_out)
        self.conv_last = nn.Sequential(
            nn.Conv2d(
                len(fpn_inplanes) * fpn_dim,
                fpn_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            ModuleHelper.BNReLU(fpn_dim, bn_type=bn_type),
            nn.Conv2d(fpn_dim, num_class, kernel_size=1),
        )
        self.norm = nn.LayerNorm(fc_dim * 4)
        self.norm1 = nn.LayerNorm(fc_dim)
        self.norm2 = nn.LayerNorm(fc_dim)
        self.norm3 = nn.LayerNorm(fc_dim)
        self.norm4 = nn.LayerNorm(fc_dim)
        self.up1 = nn.Sequential(
            *[
                nn.ConvTranspose2d(fc_dim, fc_dim, 2, 2),
                nn.GroupNorm(32, fc_dim),
                nn.GELU(),
                nn.ConvTranspose2d(fc_dim, fc_dim, 2, 2),
            ]
        )
        self.up2 = nn.ConvTranspose2d(fc_dim, fc_dim, 2, 2)
        self.up3 = nn.Identity()
        self.up4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up1.apply(self._init_weights)
        self.up2.apply(self._init_weights)
        self.up3.apply(self._init_weights)
        self.up4.apply(self._init_weights)

    def build_pyramid_features(self, x):
        b, hw, dim = x[0].size()
        ori_h = ori_w = math.sqrt(hw)
        if len(x) == 1:
            x = x[0]
            if self.feature_normalization == "global":
                x = self.norm(x)
            dim = dim // 4
            f1, f2, f3, f4 = torch.split(x, dim, dim=-1)
        else:
            f1, f2, f3, f4 = x
        if self.feature_normalization == "shared_chunk_norm1":
            # Historical CMPANet-S head: the same learned norm1 is used for all four chunks.
            f1, f2, f3, f4 = (self.norm1(f) for f in (f1, f2, f3, f4))
        f1 = f1.transpose(1, 2).reshape(-1, dim, int(ori_h), int(ori_w))
        f2 = f2.transpose(1, 2).reshape(-1, dim, int(ori_h), int(ori_w))
        f3 = f3.transpose(1, 2).reshape(-1, dim, int(ori_h), int(ori_w))
        f4 = f4.transpose(1, 2).reshape(-1, dim, int(ori_h), int(ori_w))
        f1 = self.up1(f1).contiguous()
        f2 = self.up2(f2).contiguous()
        f3 = self.up3(f3).contiguous()
        f4 = self.up4(f4).contiguous()
        return [f1, f2, f3, f4]

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_fusion(self, conv_out, segSize=None):
        if len(conv_out[0].shape) == 3 and isinstance(conv_out, list):
            conv_out = self.build_pyramid_features(conv_out)
        conv5 = conv_out[-1]
        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale, pool_conv in zip(self.ppm_pooling, self.ppm_conv):
            ppm_out.append(
                pool_conv(
                    nn.functional.interpolate(
                        pool_scale(conv5),
                        (input_size[2], input_size[3]),
                        mode="bilinear",
                        align_corners=False,
                    )
                )
            )
        ppm_out = torch.cat(ppm_out, 1)
        f = self.ppm_last_conv(ppm_out)
        fpn_feature_list = [f]
        for i in reversed(range(len(conv_out) - 1)):
            conv_x = conv_out[i]
            conv_x = self.fpn_in[i](conv_x)
            f = nn.functional.interpolate(
                f, size=conv_x.size()[2:], mode="bilinear", align_corners=False
            )
            f = conv_x + f
            fpn_feature_list.append(self.fpn_out[i](f))
        fpn_feature_list.reverse()
        output_size = fpn_feature_list[0].size()[2:]
        fusion_list = [fpn_feature_list[0]]
        for i in range(1, len(fpn_feature_list)):
            fusion_list.append(
                nn.functional.interpolate(
                    fpn_feature_list[i],
                    output_size,
                    mode="bilinear",
                    align_corners=False,
                )
            )
        fusion_out = torch.cat(fusion_list, 1)
        x = self.conv_last(fusion_out)
        if self.use_softmax:
            x = nn.functional.interpolate(
                x, size=segSize, mode="bilinear", align_corners=False
            )
            x = nn.functional.softmax(x, dim=1)
            return x
        return (x, fusion_out)

    def forward(self, conv_out, segSize=None):
        if len(conv_out[0].shape) == 3 and isinstance(conv_out, list):
            conv_out = self.build_pyramid_features(conv_out)
        conv5 = conv_out[-1]
        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale, pool_conv in zip(self.ppm_pooling, self.ppm_conv):
            ppm_out.append(
                pool_conv(
                    nn.functional.interpolate(
                        pool_scale(conv5),
                        (input_size[2], input_size[3]),
                        mode="bilinear",
                        align_corners=False,
                    )
                )
            )
        ppm_out = torch.cat(ppm_out, 1)
        f = self.ppm_last_conv(ppm_out)
        fpn_feature_list = [f]
        for i in reversed(range(len(conv_out) - 1)):
            conv_x = conv_out[i]
            conv_x = self.fpn_in[i](conv_x)
            f = nn.functional.interpolate(
                f, size=conv_x.size()[2:], mode="bilinear", align_corners=False
            )
            f = conv_x + f
            fpn_feature_list.append(self.fpn_out[i](f))
        fpn_feature_list.reverse()
        output_size = fpn_feature_list[0].size()[2:]
        fusion_list = [fpn_feature_list[0]]
        for i in range(1, len(fpn_feature_list)):
            fusion_list.append(
                nn.functional.interpolate(
                    fpn_feature_list[i],
                    output_size,
                    mode="bilinear",
                    align_corners=False,
                )
            )
        fusion_out = torch.cat(fusion_list, 1)
        x = self.conv_last(fusion_out)
        if self.use_softmax:
            x = nn.functional.interpolate(
                x, size=segSize, mode="bilinear", align_corners=False
            )
            x = nn.functional.softmax(x, dim=1)
            return x
        return (x, fusion_out)
