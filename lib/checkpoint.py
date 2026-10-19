"""Strict loading for the released historical decoder and current checkpoints."""
import hashlib
from pathlib import Path
import torch
from torch import nn
from lib.models.tools.module_helper import ModuleHelper
import lib.utils.tools.configer as configer_module
import sys


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_checkpoint(path):
    # Historical files can refer to this original module name. Load only trusted checkpoints.
    sys.modules.setdefault('configer', configer_module)
    return torch.load(path, map_location='cpu', weights_only=False)


def load_decoder(decoder, checkpoint, cfg):
    saved = load_checkpoint(checkpoint)
    state = saved['state_dict']
    if 'chg_head.norm.weight' not in state:
        if decoder.chg_head.feature_normalization != 'shared_chunk_norm1':
            raise ValueError('Historical release weights require shared_chunk_norm1.')
        decoder.chg_head.norm = nn.Identity()
    legacy_fuse = any(k.startswith('fuse_convs.') for k in state)
    if legacy_fuse:
        # Retained for exact state compatibility; these modules are unused by forward.
        params = cfg['network']['decoder']['params']
        dim = params['encoder_dim']
        decoder.fuse_convs = nn.ModuleList([
            nn.Sequential(nn.Conv2d(dim*2, dim*4, 1),
                          ModuleHelper.BNReLU(dim*4, bn_type=cfg['network']['bn_type']),
                          nn.Conv2d(dim*4, dim, 1),
                          ModuleHelper.BNReLU(dim, bn_type=cfg['network']['bn_type']))
            for _ in range(params['n_cross_layer'])
        ])
    decoder.load_state_dict(state, strict=True)
    return {'strict': True, 'tensors': len(state), 'legacy_unused_fuse_convs': legacy_fuse}
