from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import glob
import os
import cv2
import torch
import numpy as np
import random
import math
from tqdm import tqdm
from xml.dom.minidom import parse
from lib.utils.tools.logger import Logger as Log
import torch.nn.functional as F


class cd_semi_dataset(torch.utils.data.Dataset):

    def __init__(
        self,
        data_dir,
        aug_transform=None,
        img_transform=None,
        label_transform=None,
        configer=None,
        mode=None,
        **kwargs
    ):
        self.configer = configer
        self.aug_transform = aug_transform
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.num_classes = self.configer.get("data", "num_classes")
        self.dataset = kwargs["dataset"]
        self.odgt = data_dir
        self.check_path = kwargs["check_path"]
        self.transxml = self.configer.get("config", "transxml")
        self.dict = None
        if os.path.exists(self.transxml):
            _, self.dict = self.readtransxml(self.transxml)
        self.edges = None
        if self.dataset == "semi_train":
            with open(self.configer.get("data", "label_txt"), "r") as f:
                self.labeled_ids = f.readlines()
            f.close()
            with open(self.configer.get("data", "ulabel_txt"), "r") as f:
                self.unlabeled_ids = f.readlines()
            f.close()
            self.ids = (
                self.labeled_ids
                * math.ceil(len(self.unlabeled_ids) / len(self.labeled_ids))
                + self.unlabeled_ids
            )
            if self.check_path:
                self.checkpaths(self.ids, shuffle=True)
        elif self.dataset == "val":
            with open(data_dir, "r", encoding="utf-8") as f:
                self.ids = f.readlines()
            if self.configer.exists("data", "edges_val"):
                search_dir = self.configer.get("data", "edges_val")
                self.edges = glob.glob(search_dir + "/*.jpg")
                for i, pair in enumerate(self.edges):
                    self.ids[i] = self.ids[i].rstrip() + " " + pair
            f.close()
        else:
            with open(data_dir, "r", encoding="utf-8") as f:
                self.ids = f.readlines()
            if self.configer.exists("data", "edges_train"):
                search_dir = self.configer.get("data", "edges_train")
                self.edges = glob.glob(search_dir + "/*.jpg")
                for i, pair in enumerate(self.edges):
                    self.ids[i] = self.ids[i].rstrip() + " " + pair
            f.close()
            if self.check_path:
                self.checkpaths(self.ids, clampN=False)

    def checkpaths(self, lines, shuffle=True, clampN=False):
        self.ids = []
        for id, line in tqdm(enumerate(lines)):
            paths = line.rstrip().split()
            exist_flag = True
            for path in paths:
                if not os.path.exists(path):
                    print("{} is not existed".format(path))
                    exist_flag = False
                    break
            if exist_flag:
                self.ids.append(line)
        if shuffle:
            random.shuffle(self.ids)
        if clampN:
            self.ids = self.ids[:1000]
        return self.ids

    def readtransxml(self, transxml):
        path_transxml = transxml
        if not os.path.exists(path_transxml):
            Log("Error:{} is not existed.".format(path_transxml))
        transBM = parse(path_transxml)
        root = transBM.documentElement
        all_codes = root.getElementsByTagName("BM")
        all_dict = {}
        num_class = 0
        for node in all_codes:
            class_geoid_name = node.attributes["key"].value
            class_id = node.attributes["val"].value
            all_dict[int(class_geoid_name)] = int(class_id)
            if int(class_id) > num_class:
                num_class = num_class + 1
        return (num_class, all_dict)

    def __len__(self):
        return len(self.ids)


class shpcd_semi_dataset(cd_semi_dataset):

    def __init__(
        self,
        data_dir,
        aug_transform=None,
        img_transform=None,
        label_transform=None,
        configer=None,
        mode=None,
        **kwargs
    ):
        super(shpcd_semi_dataset, self).__init__(
            data_dir=data_dir,
            aug_transform=aug_transform,
            img_transform=img_transform,
            label_transform=label_transform,
            configer=configer,
            mode=mode,
            **kwargs
        )

    def __getitem__(self, idx):
        oldimage_path = self.ids[idx].rstrip().split()[0]
        oldimg = cv2.imdecode(
            np.fromfile(oldimage_path, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        newimage_path = self.ids[idx].rstrip().split()[1]
        newimg = cv2.imdecode(np.fromfile(newimage_path, dtype=np.uint8), -1)
        label_path = self.ids[idx].rstrip().split()[2]
        label = cv2.imdecode(
            np.fromfile(label_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE
        )
        if self.dict is not None:
            try:
                label_map = np.vectorize(self.dict.get)(label)
            except Exception:
                unique_val = np.unique(label)
                Log.error(unique_val)
            if None in label_map:
                pix_val = np.unique(label)
                mix_index = []
                for pix in pix_val:
                    if str(pix_val) not in self.dict.keys():
                        mix_index.append(pix)
                assert len(mix_index) == 0, "{}\n cannot find {} in transxml.".format(
                    label_path, mix_index
                )
            else:
                label = label_map
        oldimg_zero = (
            (oldimg[:, :, 0] == 0) & (oldimg[:, :, 1] == 0) & (oldimg[:, :, 2] == 0)
        )
        newimg_zero = (newimg[:, :, 0] == 0) & (newimg[:, :, 1] == 0) & (
            newimg[:, :, 2] == 0
        ) | (newimg[:, :, 0] == 1) & (newimg[:, :, 1] == 1) & (newimg[:, :, 2] == 1)
        maskzero = oldimg_zero | newimg_zero
        label[label > 0] = 1
        oldimg[maskzero, :] = 0
        newimg[maskzero, :] = 0
        label[maskzero] = 255
        if self.aug_transform is not None:
            oldimg, newimg, label = self.aug_transform(oldimg, newimg, labelmap=label)
        label = torch.from_numpy(label)
        if self.img_transform is not None:
            if isinstance(self.img_transform, list):
                oldimg = self.img_transform[0](oldimg)
                newimg = self.img_transform[1](newimg)
            else:
                oldimg = self.img_transform(oldimg)
                newimg = self.img_transform(newimg)
        if self.label_transform is not None:
            label = self.label_transform(label)
        H, W = (label.shape[-2] // 16, label.shape[-1] // 16)
        mask_chg = torch.zeros_like(label, dtype=torch.float)
        mask_chg[label > 0] = 1
        mask_patch = (
            F.interpolate(
                mask_chg.float().unsqueeze(0).unsqueeze(0), size=(H, W), mode="bilinear"
            )
            .squeeze(0)
            .squeeze(0)
        )
        mask_patch[mask_patch > 0] = 1
        if isinstance(oldimg, np.ndarray):
            oldimg = torch.from_numpy(oldimg).permute(2, 0, 1)
        if isinstance(newimg, np.ndarray):
            newimg = torch.from_numpy(newimg).permute(2, 0, 1)
        img = torch.cat([oldimg, newimg], dim=0)
        img_metas = {}
        img_metas["old_path"] = oldimage_path
        img_metas["new_path"] = newimage_path
        img_metas["label_path"] = label_path
        return (img, label, mask_patch, img_metas)

    def __len__(self):
        return len(self.ids)
