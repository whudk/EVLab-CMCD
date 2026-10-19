#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Some methods used by main methods.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import torch
from lib.utils.tools.logger import Logger as Log
from lib.utils.distributed import get_rank, is_distributed


class ModuleRunner(object):

    def __init__(self, configer):
        self.configer = configer
        self._init()

    def _init(self):
        self.configer.add(["iters"], 0)
        self.configer.add(["last_iters"], 0)
        self.configer.add(["epoch"], 0)
        self.configer.add(["last_epoch"], 0)
        self.configer.add(["max_performance"], 0.0)
        self.configer.add(["performance"], 0.0)
        self.configer.add(["max_accuracy"], 0.0)
        self.configer.add(["accuracy"], 0.0)
        self.configer.add(["min_val_loss"], 9999.0)
        self.configer.add(["val_loss"], 9999.0)
        self.half = False
        if self.configer.exists("network", "train_half"):
            self.half = self.configer.exists("network", "train_half")
        if not self.configer.exists("network", "bn_type"):
            self.configer.add(["network", "bn_type"], "torchbn")
        if self.configer.get("phase") == "train":
            assert (
                len(self.configer.get("gpu")) > 1
                or self.configer.get("network", "bn_type") == "torchbn"
            )
        Log.info("BN Type is {}.".format(self.configer.get("network", "bn_type")))

    def to_device(self, *params, force_list=False):
        moved = [p.cuda() for p in params]
        return moved if force_list or len(moved) != 1 else moved[0]

    def _make_parallel(self, net):
        return net

    def restart_from_checkpoint(self, ckp_path, run_variables=None, **kwargs):
        checkpoint = torch.load(ckp_path, map_location="cpu")
        for key, model in kwargs.items():
            if key not in checkpoint:
                raise ValueError(f"Pretrained checkpoint must contain {key!r}.")
            state = {k.removeprefix("module."): v for k, v in checkpoint[key].items()}
            model.load_state_dict(state, strict=True)
            Log.info(f"Loaded pretrained {key} from {ckp_path}")

    def load_net(self, net, net_path=None):
        net_path = net_path or self.configer.get("network", "resume")
        if net_path:
            if hasattr(net, "chg_head"):
                from lib.checkpoint import load_decoder
                load_decoder(net, net_path, self.configer.to_dict())
            else:
                checkpoint = torch.load(net_path, map_location="cpu")
                state = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
                state = {k.removeprefix("module."): v for k, v in state.items()}
                net.load_state_dict(state, strict=True)
            Log.info(f"Loaded decoder weights from {net_path}; optimizer starts fresh.")
        return self.to_device(net.float())

    def save_net(self, net, save_mode="iters", ckpt_name=None):
        if is_distributed() and get_rank() != 0:
            return
        state = {"config_dict": self.configer.to_dict(), "state_dict": net.state_dict()}
        if self.configer.get("checkpoints", "checkpoints_root") is None:
            checkpoints_dir = os.path.join(
                self.configer.get("project_dir"),
                self.configer.get("checkpoints", "checkpoints_dir"),
            )
        else:
            checkpoints_dir = os.path.join(
                self.configer.get("checkpoints", "checkpoints_root"),
                self.configer.get("checkpoints", "checkpoints_dir"),
            )
        if not os.path.exists(checkpoints_dir):
            os.makedirs(checkpoints_dir)
        if ckpt_name is None:
            latest_name = "{}_latest.pth".format(
                self.configer.get("checkpoints", "checkpoints_name")
            )
        else:
            latest_name = "{}_latest.pth".format(ckpt_name)
        torch.save(state, os.path.join(checkpoints_dir, latest_name))
        if save_mode == "performance":
            if self.configer.get("performance") >= self.configer.get("max_performance"):
                if ckpt_name is None:
                    latest_name = "{}_max_performance.pth".format(
                        self.configer.get("checkpoints", "checkpoints_name")
                    )
                else:
                    latest_name = "{}_max_performance.pth".format(ckpt_name)
                torch.save(state, os.path.join(checkpoints_dir, latest_name))
                self.configer.update(
                    ["max_performance"], self.configer.get("performance")
                )
        elif save_mode == "val_loss":
            if self.configer.get("val_loss") < self.configer.get("min_val_loss"):
                if ckpt_name is None:
                    latest_name = "{}_min_loss.pth".format(
                        self.configer.get("checkpoints", "checkpoints_name")
                    )
                else:
                    latest_name = "{}_min_loss.pth".format(ckpt_name)
                torch.save(state, os.path.join(checkpoints_dir, latest_name))
                self.configer.update(["min_val_loss"], self.configer.get("val_loss"))
        elif save_mode == "iters":
            if self.configer.get("iters") - self.configer.get(
                "last_iters"
            ) >= self.configer.get("checkpoints", "save_iters"):
                if ckpt_name is None:
                    latest_name = "{}_iters{}.pth".format(
                        self.configer.get("checkpoints", "checkpoints_name"),
                        self.configer.get("iters"),
                    )
                else:
                    latest_name = "{}_iters{}.pth".format(
                        ckpt_name, self.configer.get("iters")
                    )
                torch.save(state, os.path.join(checkpoints_dir, latest_name))
                self.configer.update(["last_iters"], self.configer.get("iters"))
        elif save_mode == "epoch":
            if self.configer.get("epoch") - self.configer.get(
                "last_epoch"
            ) >= self.configer.get("checkpoints", "save_epoch"):
                if ckpt_name is None:
                    latest_name = "{}_epoch{}.pth".format(
                        self.configer.get("checkpoints", "checkpoints_name"),
                        self.configer.get("epoch"),
                    )
                else:
                    latest_name = "{}_epoch{}.pth".format(
                        ckpt_name, self.configer.get("epoch")
                    )
                torch.save(state, os.path.join(checkpoints_dir, latest_name))
        else:
            Log.error("Metric: {} is invalid.".format(save_mode))
            exit(1)

    def get_lr_from_scheduler(self, optimizer, scheduler):
        it = self.configer.get("iters")
        lr = scheduler[min(it, len(scheduler) - 1)]
        for group in optimizer.param_groups:
            group["lr"] = lr
        return lr
