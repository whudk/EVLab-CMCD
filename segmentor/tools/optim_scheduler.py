#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com), Lang Huang, Rainbowsecret
# Some methods used by main methods.

import numpy as np
from torch.optim import AdamW


class OptimScheduler:

    def __init__(self, configer):
        self.configer = configer

    def init_optimizer(self, net_params):
        if self.configer.get("optim", "optim_method") != "adamw":
            raise ValueError("Only the configured AdamW optimizer is supported.")
        return (
            AdamW(
                net_params,
                lr=self.configer.get("lr", "base_lr"),
                **self.configer.get("optim", "adamw")
            ),
            None,
        )

    def cosine_scheduler(
        self,
        base_value,
        final_value,
        epochs,
        niter_per_ep,
        warmup_epochs=0,
        start_warmup_value=0,
    ):
        warmup_iters = warmup_epochs * niter_per_ep
        warmup = np.linspace(start_warmup_value, base_value, warmup_iters)
        iters = np.arange(epochs * niter_per_ep - warmup_iters)
        schedule = final_value + 0.5 * (base_value - final_value) * (
            1 + np.cos(np.pi * iters / len(iters))
        )
        return np.concatenate((warmup, schedule))
