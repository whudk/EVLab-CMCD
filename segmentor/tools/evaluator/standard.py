import numpy as np
import torch
from .base import _BaseEvaluator


class StandardEvaluator(_BaseEvaluator):

    def _init_running_scores(self):
        from lib.metrics.running_score import RunningScore

        self.running_scores = {"seg": RunningScore(self.configer, self.num_classes)}
        self.save_net_main_key = "seg"
        self.save_net_metric = "miou"

    def update_score_seg(self, outputs, targets):
        pred = outputs.cpu().numpy()
        target = targets.cpu().numpy()
        self.running_scores["seg"].update(pred, target)

    def evaluation_object(self, preds, targets):
        from skimage import measure

        targets = targets.clone()
        preds = preds.clone()
        preds[targets == -1] = 0
        targets[targets == -1] = 0
        if isinstance(preds, torch.Tensor):
            pred1 = preds.cpu().numpy().astype(np.uint8)
        if isinstance(targets, torch.Tensor):
            gt = targets.cpu().numpy().astype(np.uint8)
        bs = preds.shape[0]
        chk_n = 0
        ture_n = 0
        pred_n = 0
        gt_n = 0
        for b in range(bs):
            preds_binary = pred1[b]
            targets_binary = gt[b]
            preds_labeled, m = measure.label(preds_binary, return_num=True)
            targets_labeled, n = measure.label(targets_binary, return_num=True)
            for i in range(1, n + 1):
                mask = np.zeros_like(preds_labeled)
                mask[targets_labeled == i] = 1
                if np.sum(preds_labeled * mask) > 0:
                    chk_n = chk_n + 1
            for j in range(1, m + 1):
                mask = np.zeros_like(targets_labeled)
                mask[preds_labeled == j] = 1
                if np.sum(targets_labeled * mask) > 0:
                    ture_n = ture_n + 1
            pred_n = pred_n + m
            gt_n = gt_n + n
        return (pred_n, gt_n, chk_n, ture_n)
