import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_, to_2tuple
import math
import einops


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class channelsAttention(nn.Module):

    def __init__(self, in_dim):
        super(channelsAttention, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 16, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_dim // 16, out_channels=in_dim, kernel_size=1),
        )
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avgpool(x))
        max_out = self.mlp(self.maxpool(x))
        out = avg_out + max_out
        attention = self.sigmod(out)
        return attention * x


class spatialAttention(nn.Module):

    def __init__(self):
        super(spatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=1)
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat((avg_out, max_out), dim=1)
        out = self.conv1(out)
        attention = self.sigmod(out)
        return attention * x


class CAM_PAM(nn.Module):

    def __init__(self, dim):
        super(CAM_PAM, self).__init__()
        self.att = nn.Sequential(channelsAttention(dim), spatialAttention())

    def forward(self, x):
        if len(x.shape) == 3:
            B, N, C = x.size()
            h = int(math.sqrt(N))
            x = einops.rearrange(x, "b (h w) c  -> b c h w", h=h)
        x = x + self.att(x)
        return x


class Mlp(nn.Module):

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop_path = DropPath(drop) if drop > 0.0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        if len(x.shape) == 4:
            _, _, H, W = x.size()
            x = einops.rearrange(x, "b c h w -> b (h w) c")
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop_path(x)
        x = self.fc2(x)
        x = self.drop_path(x)
        return x


class Attention(nn.Module):

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** (-0.5)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (qkv[0], qkv[1], qkv[2])
        attn = q @ k.transpose(-2, -1) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return (x, attn)


class standardCrossAttention(nn.Module):

    def __init__(self, dim, num_heads):
        super(standardCrossAttention, self).__init__()
        self.num_heads = num_heads
        self.attention = nn.MultiheadAttention(dim, num_heads)

    def forward(self, query, key, value):
        attn_output, _ = self.attention(query, key, value)
        return attn_output + query


class Channel_wise_Cross_Modality(nn.Module):

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        window_size=8,
        qk_scale=None,
        attn_drop=0.0,
        act_layer=nn.GELU,
        proj_drop=0.0,
        feats_exchange="CCMAT",
    ):
        super().__init__()
        self.channel_wise_attn = Cross_ChannelsAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            window_size=window_size,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        self.cross_Modality = CrossAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            window_size=window_size,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        self.norm = nn.LayerNorm(dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.drop_path = DropPath(attn_drop) if attn_drop > 0.0 else nn.Identity()
        self.mlp1 = Mlp(
            in_features=dim,
            hidden_features=dim * 4,
            out_features=dim,
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.mlp2 = Mlp(
            in_features=dim,
            hidden_features=dim * 4,
            out_features=dim,
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.feats_exchange = feats_exchange
        if self.feats_exchange not in ["CMT", "CCT", "CCMAT", "NONE"]:
            raise Exception(
                "interaction type {} is not supported".format(self.feats_exchange)
            )

    def forward(self, x1, x2):
        if self.feats_exchange == "NONE":
            return (x1, x2)
        if self.feats_exchange == "CMT":
            y1, y2, _, _ = self.cross_Modality(self.norm(x1), self.norm(x2))
        elif self.feats_exchange == "CCT":
            y1, y2, _ = self.channel_wise_attn(self.norm(x1), self.norm(x2))
        elif self.feats_exchange == "CCMAT":
            y1, y2, _, _ = self.cross_Modality(self.norm(x1), self.norm(x2))
            y1, y2, _ = self.channel_wise_attn(y1, y2)
        x1 = x1 + self.drop_path(y1)
        x1 = x1 + self.drop_path(self.mlp1(self.norm1(x1)))
        x2 = x2 + self.drop_path(y2)
        x2 = x2 + self.drop_path(self.mlp2(self.norm2(x2)))
        return (x1, x2, _)


class ChannelsAttention(nn.Module):

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        window_size=8,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super(ChannelsAttention, self).__init__()
        window_size = to_2tuple(window_size)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.num_heads = num_heads
        assert dim % num_heads == 0
        head_dim = dim // num_heads
        self.scale = head_dim ** (-0.5)
        self.window_size = window_size
        self.attn_drop = DropPath(attn_drop) if attn_drop > 0.0 else nn.Identity()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = DropPath(proj_drop) if proj_drop > 0.0 else nn.Identity()
        self.parm = nn.Parameter(torch.zeros(1))
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        q = q * self.scale
        q = einops.rearrange(q, "b n (h1 c1) -> b h1 n c1", h1=self.num_heads)
        k = einops.rearrange(k, "b n (h1 c1) -> b h1 n c1", h1=self.num_heads)
        v = einops.rearrange(v, "b n (h1 c1) -> b h1 n c1", h1=self.num_heads)
        attn = q * k
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        y = attn * v
        y = einops.rearrange(y, "b h n c1 -> b n (h c1)")
        y = self.proj(y)
        y = self.proj_drop(y)
        x = y * self.parm + x
        return (x, attn)


class Cross_ChannelsAttention(nn.Module):

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        window_size=8,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        window_size = to_2tuple(window_size)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.num_heads = num_heads
        assert dim % num_heads == 0
        head_dim = dim // num_heads
        self.scale = head_dim ** (-0.5)
        self.window_size = window_size
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.parm = nn.Parameter(torch.zeros(1))
        self.norm = nn.LayerNorm(dim)

    def forward(self, x1, x2):
        B, N, C = x1.size()
        q = self.q(x1)
        k = self.k(x2)
        v1 = self.v(x1)
        v2 = self.v(x2)
        q = q * self.scale
        q = einops.rearrange(q, "b n (h1 c1) -> b h1 n c1", h1=self.num_heads)
        k = einops.rearrange(k, "b n (h1 c1) -> b h1 n c1", h1=self.num_heads)
        v1 = einops.rearrange(v1, "b n (h1 c1) -> b h1 n c1", h1=self.num_heads)
        v2 = einops.rearrange(v2, "b n (h1 c1) -> b h1 n c1", h1=self.num_heads)
        attn = q * k
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x1 = attn * v1
        x1 = einops.rearrange(x1, "b h n c1 -> b n (h c1)")
        x1 = self.proj(x1)
        x1 = self.proj_drop(x1)
        x2 = attn * v2
        x2 = einops.rearrange(x2, "b h n c1 -> b n (h c1)")
        x2 = self.proj(x2)
        x2 = self.proj_drop(x2)
        return (x1, x2, attn)


class CrossAttention_Single(nn.Module):

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.num_heads = num_heads
        assert dim % num_heads == 0
        head_dim = dim // num_heads
        self.scale = head_dim ** (-0.5)
        self.attn_drop = DropPath(attn_drop) if attn_drop > 0.0 else nn.Identity()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = DropPath(proj_drop) if proj_drop > 0.0 else nn.Identity()
        self.parm = nn.Parameter(torch.zeros(1))
        self.norm = nn.LayerNorm(dim)

    def forward(self, x1, x2):
        B, N, C = x1.size()
        q = self.q(x1)
        k = self.k(x2)
        v = self.v(x1)
        q = q * self.scale
        q = einops.rearrange(q, "b n  (h1  c) -> b  h1 n c", h1=self.num_heads)
        k = einops.rearrange(k, "b n  (h1  c) -> b  h1 n c", h1=self.num_heads)
        v = einops.rearrange(v, "b n  (h1  c) -> b  h1 n c", h1=self.num_heads)
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return (x, attn)


class CrossAttention(nn.Module):

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        window_size=8,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        window_size = to_2tuple(window_size)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.num_heads = num_heads
        assert dim % num_heads == 0
        head_dim = dim // num_heads
        self.scale = head_dim ** (-0.5)
        self.window_size = window_size
        self.attn_drop = DropPath(attn_drop) if attn_drop > 0.0 else nn.Identity()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = DropPath(proj_drop) if proj_drop > 0.0 else nn.Identity()
        self.parm = nn.Parameter(torch.zeros(1))
        self.norm = nn.LayerNorm(dim)

    def forward(self, x1, x2):
        B, N, C = x1.size()
        q1, q2 = (self.q(x1), self.q(x2))
        k1, k2 = (self.k(x1), self.k(x2))
        v1, v2 = (self.v(x1), self.v(x2))
        q1, q2 = (q1 * self.scale, q2 * self.scale)
        q1 = einops.rearrange(q1, "b n  (h1  c) -> b  h1 n c", h1=self.num_heads)
        k1 = einops.rearrange(k1, "b n  (h1  c) -> b  h1 n c", h1=self.num_heads)
        v1 = einops.rearrange(v1, "b n  (h1  c) -> b  h1 n c", h1=self.num_heads)
        q2 = einops.rearrange(q2, "b n  (h1  c) -> b  h1 n c", h1=self.num_heads)
        k2 = einops.rearrange(k2, "b n  (h1  c) -> b  h1 n c", h1=self.num_heads)
        v2 = einops.rearrange(v2, "b n  (h1  c) -> b  h1 n c", h1=self.num_heads)
        attn1 = q1 @ k2.transpose(-2, -1)
        attn1 = attn1.softmax(dim=-1)
        attn1 = self.attn_drop(attn1)
        attn2 = q2 @ k1.transpose(-2, -1)
        attn2 = attn2.softmax(dim=-1)
        attn2 = self.attn_drop(attn2)
        x1 = (attn1 @ v1).transpose(1, 2).reshape(B, N, C)
        x1 = self.proj(x1)
        x1 = self.proj_drop(x1)
        x2 = (attn2 @ v2).transpose(1, 2).reshape(B, N, C)
        x2 = self.proj(x2)
        x2 = self.proj_drop(x2)
        return (x1, x2, attn1, attn2)


class CrossAttention_Block(nn.Module):

    def __init__(
        self,
        dim,
        num_heads,
        window_size=8,
        mlp_ratio=4.0,
        depth=1,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        feats_fusion="TBAM",
        feats_exchange=None,
    ):
        super(CrossAttention_Block, self).__init__()
        self.feats_exchange = feats_exchange
        self.drop_path = nn.Identity()
        self.norm = nn.LayerNorm(dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm_fuse = nn.LayerNorm(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=dim * 4,
            out_features=dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.mlp2 = Mlp(
            in_features=dim * 2,
            hidden_features=dim * 4,
            out_features=dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.init_attn(
            dim, num_heads, window_size, depth, qkv_bias, qk_scale, drop, attn_drop
        )
        self.feats_fusion = feats_fusion
        if self.feats_fusion == "TBAM":
            self.attn = Attention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
            )
            self.ch_attn = ChannelsAttention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
            )
        elif self.feats_fusion == "SCT":
            self.fusion = standardCrossAttention(num_heads=num_heads, dim=dim)
        elif self.feats_fusion == "diff":
            self.fusion = nn.PairwiseDistance(2)
        elif self.feats_fusion == "concat":
            self.fusion = Mlp(
                in_features=dim * 2,
                hidden_features=dim * 4,
                out_features=dim,
                act_layer=act_layer,
                drop=drop,
            )
        elif self.feats_fusion == "cross":
            self.fusion = CrossAttention_Single(dim=dim, num_heads=24)
        elif self.feats_fusion == "CBAM":
            self.fusion = CAM_PAM(dim=dim)
        else:
            raise Exception

    def init_attn(
        self,
        dim,
        num_heads,
        window_size=8,
        depth=1,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        if self.feats_exchange is None:
            self.blocks = None
            return
        self.blocks = nn.ModuleList(
            [
                Channel_wise_Cross_Modality(
                    dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    attn_drop=attn_drop,
                    proj_drop=drop,
                    feats_exchange=self.feats_exchange,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x1, x2, return_attention=False):
        if self.blocks is not None:
            for i, blk in enumerate(self.blocks):
                x1, x2 = blk(x1, x2)[:2]
        if self.feats_fusion == "TBAM":
            x = self.mlp2(torch.cat((x1, x2), dim=-1))
            x, ch_attn = self.ch_attn(self.norm(x))
            y, attn = self.attn(self.norm_fuse(x))
            if return_attention:
                return attn
            x = x + self.drop_path(y)
            x = x + self.drop_path(self.mlp(self.norm1(x)))
        elif self.feats_fusion == "diff":
            x = self.fusion(x1, x2)
        elif self.feats_fusion == "concat":
            x = self.fusion(torch.cat((x1, x2), dim=-1))
        elif self.feats_fusion == "SCT":
            x = self.fusion(self.norm(x1), self.norm1(x2), self.norm1(x2))
        elif self.feats_fusion == "CBAM":
            x = self.mlp2(torch.cat((x1, x2), dim=-1))
            x = self.fusion(x)
            x = einops.rearrange(x, "b c h w -> b (h w) c")
        return x
