import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.models.layers import trunc_normal_
from lib.models.backbones.backbone_selector import BackboneSelector
from lib.models.decoder.BuildHead import BuildHead
from lib.models.nets.attention import Mlp, CrossAttention_Block
import torch.distributed as dist
from lib.models.modules.projection import ProjectionHead
import einops


class Decoder_chg(nn.Module):

    def __init__(
        self,
        configer=None,
        use_pyrid=False,
        encoder_dim=768,
        n_cross_head=6,
        n_cross_layer=4,
        **head_params
    ):
        super().__init__()
        self.configer = configer
        self.bn_type = "torchsyncbn" if dist.is_initialized() else "torchbn"
        self.num_classes = self.configer.get("data", "num_classes")
        self.use_pyrid = use_pyrid
        self.mode = self.configer.get("train", "mode")
        chg_head_dict = self.configer.get("network", "decoder")["head"]
        fc_dim = encoder_dim
        self.fusion_type = chg_head_dict["fusion"]
        embed_dim = chg_head_dict["params"]["embed_dim"]
        self.embed_layer = Mlp(
            fc_dim * 4, hidden_features=fc_dim, out_features=embed_dim, drop=0.0
        )
        self.feats_interacts = []
        self.feats_interacts = CrossAttention_Block(
            fc_dim,
            num_heads=n_cross_head,
            window_size=1,
            depth=n_cross_layer,
            mlp_ratio=4.0,
            qkv_bias=False,
            qk_scale=None,
            drop=0.0,
            attn_drop=0.0,
        )
        if self.configer.exists("contrast"):
            self.r = self.configer.get("contrast", "memory_size")
            self.register_buffer(
                "segment_queue", torch.randn(self.num_classes, self.r, embed_dim)
            )
            self.segment_queue = nn.functional.normalize(self.segment_queue, p=2, dim=2)
            self.register_buffer(
                "segment_queue_ptr", torch.zeros(self.num_classes, dtype=torch.long)
            )
            self.register_buffer(
                "pixel_queue", torch.randn(self.num_classes, self.r, embed_dim)
            )
            self.pixel_queue = nn.functional.normalize(self.pixel_queue, p=2, dim=2)
            self.register_buffer(
                "pixel_queue_ptr", torch.zeros(self.num_classes, dtype=torch.long)
            )
        self.chg_head = BuildHead(configer).build_head(
            name=chg_head_dict["name"], **chg_head_dict["params"]
        )
        if use_pyrid:
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
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def build_pyramid_features(self, x):
        b, hw, dim = x[0].size()
        ori_h = ori_w = math.sqrt(hw)
        f1, f2, f3, f4 = x
        f1 = self.norm1(f1).transpose(1, 2).reshape(-1, dim, int(ori_h), int(ori_w))
        f2 = self.norm1(f2).transpose(1, 2).reshape(-1, dim, int(ori_h), int(ori_w))
        f3 = self.norm1(f3).transpose(1, 2).reshape(-1, dim, int(ori_h), int(ori_w))
        f4 = self.norm1(f4).transpose(1, 2).reshape(-1, dim, int(ori_h), int(ori_w))
        f1 = self.up1(f1).contiguous()
        f2 = self.up2(f2).contiguous()
        f3 = self.up3(f3).contiguous()
        f4 = self.up4(f4).contiguous()
        return [f1, f2, f3, f4]


class CDHead_shp_to_img(Decoder_chg):

    def __init__(
        self,
        configer=None,
        use_pyrid=False,
        feats_exchange="cc",
        feats_fusion="cc",
        encoder_dim=768,
        n_cross_head=6,
        n_cross_layer=4,
        patch_out_dim=8192,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__(
            configer=configer,
            use_pyrid=use_pyrid,
            encoder_dim=encoder_dim,
            n_cross_head=n_cross_head,
            n_cross_layer=n_cross_layer,
        )
        self.shp_encoder = BackboneSelector(configer).get_backbone(
            self.configer.get("network", "shp_encoder")["name"],
            arch=self.configer.get("network", "shp_encoder")["name"],
            net_params=self.configer.get("network", "shp_encoder")["params"],
        )
        encoder_dim_shp = self.shp_encoder.embed_dim
        encoder_dim_img = encoder_dim
        self.norm_last = nn.LayerNorm(encoder_dim_shp)
        self.embed_img = Mlp(encoder_dim_img * 4, encoder_dim_img * 4, encoder_dim)
        self.norm_img = nn.LayerNorm(encoder_dim)
        self.embed_shp = Mlp(encoder_dim_shp * 4, encoder_dim_shp * 4, encoder_dim)
        self.norm_shp = nn.LayerNorm(encoder_dim)
        self.patch_head = nn.Sequential(
            Mlp(
                encoder_dim_shp,
                hidden_features=encoder_dim_shp * 4,
                out_features=encoder_dim_shp,
            ),
            nn.LayerNorm(encoder_dim_shp),
            nn.Linear(encoder_dim_shp, patch_out_dim),
        )
        self.feats_interacts = CrossAttention_Block(
            encoder_dim,
            num_heads=n_cross_head,
            depth=n_cross_layer,
            window_size=1,
            mlp_ratio=4.0,
            qkv_bias=False,
            qk_scale=None,
            drop=proj_drop,
            attn_drop=attn_drop,
            feats_fusion=feats_fusion,
            feats_exchange=feats_exchange,
        )
        self.proj_head = ProjectionHead(
            dim_in=1024, proj_dim=512, bn_type=self.configer.get("network", "bn_type")
        )
        self.aux_head = nn.Linear(encoder_dim, self.num_classes)

    def forward(
        self,
        shp_img,
        feats_right,
        targets=None,
        mask=None,
        with_embed=False,
        is_eval=False,
        return_attention=False,
    ):
        feats_last = self.shp_encoder(shp_img, mask=mask)
        feats_left = self.shp_encoder.get_feats(shp_img)
        feats_left = [feats[:, 1:, :] for feats in feats_left]
        embed_algin = self.patch_head(feats_last[:, 1:, :])
        feats_left = torch.cat(feats_left, dim=-1)
        feats_right = torch.cat(feats_right, dim=-1)
        feats_left = self.norm_shp(self.embed_shp(feats_left))
        feats_right = self.norm_img(self.embed_img(feats_right))
        conv_out = []
        if return_attention is not False:
            attn = self.feats_interacts(feats_left, feats_right, return_attention)
            return {"attn": attn}
        conv_out.append(self.feats_interacts(feats_left, feats_right))
        if self.use_pyrid:
            conv_out = self.build_pyramid_features(conv_out)
            fusion_feats = []
            for idx in range(len(conv_out)):
                feats = F.interpolate(
                    conv_out[idx],
                    size=conv_out[0].shape[2:],
                    mode="bilinear",
                    align_corners=True,
                )
                fusion_feats.append(feats)
            fusion_feats = torch.cat(fusion_feats, dim=1)
        else:
            fusion_feats = torch.cat(conv_out, dim=-1)
        if len(fusion_feats.shape) == 4:
            fusion_feats = einops.rearrange(
                fusion_feats,
                "b c h w -> b (h w) c",
                h=fusion_feats.shape[2],
                w=fusion_feats.shape[3],
            )
        out, fusion = self.chg_head(conv_out)
        aux_out, embed_feats = (None, None)
        if with_embed is True:
            embed_feats = self.proj_head(fusion)
            if len(embed_feats.shape) == 3:
                embed_feats = einops.rearrange(
                    embed_feats,
                    "b (h w) c -> b c h w",
                    h=int(math.sqrt(embed_feats.shape[1])),
                )
        if with_embed and is_eval is not True:
            return {
                "pred": out,
                "pred_aux": aux_out,
                "embed": embed_feats,
                "embed_algin": embed_algin,
                "key": embed_feats.detach(),
                "lb_key": targets.detach(),
                "fusion": fusion,
            }
        else:
            return {
                "pred": out,
                "pred_aux": aux_out,
                "embed": embed_feats,
                "embed_algin": embed_algin,
                "key": embed_feats,
                "fusion": fusion,
            }
