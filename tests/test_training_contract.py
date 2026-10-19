import copy
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
import torch

from main import build_parser, load_config, validate_config
from lib.datasets.data_loader import DataLoader
from lib.loss.loss_helper import FSOhemCELoss
from lib.utils.tools.configer import Configer
from segmentor.tools.evaluator.standard import StandardEvaluator
from segmentor.tools.module_runner import ModuleRunner


class TrainingContract(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="cmpanet_contract_")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        image = np.full((64, 64, 3), 80, dtype=np.uint8)
        label = np.zeros((64, 64), dtype=np.uint8)
        label[16:32, 16:32] = 255
        paths = [self.root / name for name in ("left.png", "right.png", "label.png")]
        for path, pixels in zip(paths, (image, image, label)):
            cv2.imencode(".png", pixels)[1].tofile(path)
        line = " ".join(map(str, paths))
        train, val, weight = [
            self.root / name for name in ("train.txt", "val.txt", "teacher.pth")
        ]
        train.write_text("\n".join([line] * 4), encoding="utf-8")
        val.write_text(line, encoding="utf-8")
        weight.touch()  # validate_config checks paths, not checkpoint contents.
        args = build_parser().parse_args(
            [
                "--suptxt",
                str(train),
                "--valtxt",
                str(val),
                "--pretrained",
                str(weight),
                "--base_lr",
                "0.00003",
                "--test_interval",
                "500",
                "--patch_algin",
                "1",
                "--gpu",
                "0",
            ]
        )
        self.cfg = validate_config(load_config(args))

    def test_cli_and_singleton_batch_policy(self):
        self.assertEqual(self.cfg["lr"]["base_lr"], 0.00003)
        self.assertEqual(self.cfg["solver"]["test_interval"], 500)
        self.assertTrue(self.cfg["loss"]["patch_algin"])
        self.assertTrue(self.cfg["data"]["drop_last"])
        self.assertEqual(self.cfg["data"]["unsuptxt"], self.cfg["data"]["valtxt"])
        with self.assertRaises(ValueError):
            bad = copy.deepcopy(self.cfg)
            bad["gpu"] = [0, 1]
            validate_config(bad)

    def test_augmented_mask_shape_and_full_validation(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["train_trans"]["trans_seq"] = ["resize"]
        cfg["train_trans"]["resize"] = {"target_size": [32, 32]}
        cfg["val_trans"]["resize"] = {"target_size": [32, 32]}
        train, unsup, val = DataLoader(Configer(config_dict=cfg)).get_semi_trainloader()
        self.assertEqual(len(train), 1)  # 4 samples, batch 3, drop singleton.
        self.assertEqual(len(val), 1)  # 1 validation sample must not be dropped.
        images, labels, masks, _ = next(iter(train))
        self.assertEqual(images.shape, (3, 6, 32, 32))
        self.assertEqual(labels.shape, (3, 32, 32))
        self.assertEqual(masks.shape, (3, 2, 2))
        self.assertIn(1, labels.unique().tolist())  # source 255 means change.

    def test_object_components_and_ignore_are_not_mutated(self):
        evaluator = StandardEvaluator(Configer(config_dict=self.cfg), None)
        target = torch.zeros((1, 8, 8), dtype=torch.long)
        target[:, 1:3, 1:3] = 1
        target[:, 5:7, 5:7] = 1
        target[:, 0, 0] = -1
        pred = target.clamp_min(0)
        before = target.clone()
        self.assertEqual(evaluator.evaluation_object(pred, target), (2, 2, 2, 2))
        self.assertTrue(torch.equal(target, before))

    def test_zero_ohem_loss_remains_differentiable(self):
        criterion = FSOhemCELoss(Configer(config_dict=self.cfg))
        logits = torch.zeros(1, 2, 2, 2, requires_grad=True)
        target = torch.full((1, 2, 2), -1, dtype=torch.long)
        loss = criterion(logits, target)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(logits.grad)

    def test_scheduler_endpoint(self):
        configer = Configer(config_dict=self.cfg)
        runner = ModuleRunner(configer)
        configer.update(["iters"], 2)
        parameter = torch.nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.AdamW([parameter])
        self.assertEqual(runner.get_lr_from_scheduler(optimizer, [0.1, 0.01]), 0.01)


if __name__ == "__main__":
    unittest.main()
