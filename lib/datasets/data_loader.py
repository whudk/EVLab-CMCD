##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Donny You, RainbowSecret, JingyiXie
## Microsoft Research
## yuyua@microsoft.com
## Copyright (c) 2019
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
from torch.utils import data
import lib.datasets.tools.transforms as trans
from lib.datasets.loader.semi_loader import shpcd_semi_dataset
from lib.utils.distributed import get_world_size, is_distributed

LOADER_DICT = {"shpcd_semi_dataset": shpcd_semi_dataset}


class DataLoader(object):

    def __init__(self, configer):
        self.configer = configer
        from lib.datasets.tools import cv2_aug_transform_chg

        self.aug_train_transform = cv2_aug_transform_chg.CV2AugCompose_CHG(
            self.configer, split="train"
        )
        self.aug_val_transform = cv2_aug_transform_chg.CV2AugCompose_CHG(
            self.configer, split="val"
        )
        self.img_transform = []
        assert self.configer.exists("normalize", "left") and self.configer.exists(
            "normalize", "right"
        ), "check normalize in config.json, normalize of left,right must be in config.json"
        left_transform = trans.Compose(
            [
                trans.ToTensor(),
                trans.Normalize(
                    div_value=self.configer.get("normalize", "left")["div_value"],
                    mean=self.configer.get("normalize", "left")["mean"],
                    std=self.configer.get("normalize", "left")["std"],
                ),
            ]
        )
        self.img_transform.append(left_transform)
        right_transform = trans.Compose(
            [
                trans.ToTensor(),
                trans.Normalize(
                    div_value=self.configer.get("normalize", "right")["div_value"],
                    mean=self.configer.get("normalize", "right")["mean"],
                    std=self.configer.get("normalize", "right")["std"],
                ),
            ]
        )
        self.img_transform.append(right_transform)
        self.label_transform = trans.Compose([trans.ToLabel(), trans.ReLabel(255, -1)])

    def get_valloader(self):
        """Evaluation without requiring training data or a training manifest."""
        dataset = shpcd_semi_dataset(
            data_dir=self.configer.get("data", "valtxt"), dataset="val",
            aug_transform=self.aug_val_transform, img_transform=self.img_transform,
            label_transform=self.label_transform, configer=self.configer, check_path=False,
        )
        return data.DataLoader(dataset, batch_size=self.configer.get("val", "batch_size"),
                               shuffle=False, drop_last=False, pin_memory=True,
                               num_workers=self.configer.get("data", "workers"))

    def get_semi_trainloader(self, use_collate=True):
        loader = self.configer.get("train", "loader")
        if loader not in LOADER_DICT.keys():
            raise Exception("loader method of {} is not supported.".format(loader))
        klass = LOADER_DICT[loader]
        split = "train"
        kwargs = dict(
            dataset=None,
            aug_transform=(
                self.aug_train_transform if split == "train" else self.aug_val_transform
            ),
            img_transform=self.img_transform,
            label_transform=self.label_transform,
            configer=self.configer,
            check_path=self.configer.get("data", "check_path"),
        )
        kwargs["dataset"] = "train"
        labeled_dataset = klass(data_dir=self.configer.get("data", "suptxt"), **kwargs)
        kwargs["dataset"] = "val"
        ulabeled_dataset = klass(
            data_dir=self.configer.get("data", "unsuptxt"), **kwargs
        )
        kwargs["aug_transform"] = self.aug_val_transform
        kwargs["dataset"] = "val"
        val_dataset = klass(data_dir=self.configer.get("data", "valtxt"), **kwargs)
        if is_distributed():
            sampler_labeled = torch.utils.data.distributed.DistributedSampler(
                labeled_dataset
            )
            sampler_ulabeled = torch.utils.data.distributed.DistributedSampler(
                ulabeled_dataset
            )
            sampler_val = torch.utils.data.distributed.DistributedSampler(val_dataset)
        else:
            sampler_labeled = sampler_ulabeled = sampler_val = None
        trainloader_labeled = data.DataLoader(
            labeled_dataset,
            batch_size=self.configer.get("train", "batch_size") // get_world_size(),
            pin_memory=True,
            num_workers=self.configer.get("data", "workers") // get_world_size(),
            sampler=sampler_labeled,
            shuffle=sampler_labeled is None,
            drop_last=self.configer.get("data", "drop_last"),
        )
        trainloader_ulabeled = data.DataLoader(
            ulabeled_dataset,
            batch_size=self.configer.get("val", "batch_size") // get_world_size(),
            pin_memory=True,
            num_workers=self.configer.get("data", "workers") // get_world_size(),
            sampler=sampler_ulabeled,
            shuffle=sampler_ulabeled is None,
            drop_last=False,
        )
        valloader = data.DataLoader(
            val_dataset,
            batch_size=self.configer.get("val", "batch_size") // get_world_size(),
            pin_memory=True,
            num_workers=self.configer.get("data", "workers") // get_world_size(),
            sampler=sampler_val,
            shuffle=False,
            drop_last=False,
        )
        return (trainloader_labeled, trainloader_ulabeled, valloader)
