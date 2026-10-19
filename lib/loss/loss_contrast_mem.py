from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from abc import ABC
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.loss.loss_helper import FSOhemCELoss
from lib.utils.tools.logger import Logger as Log

LOSS_DICT = {"fs_ohemce_loss": FSOhemCELoss}


class PixelContrastLoss(nn.Module, ABC):

    def __init__(self, configer):
        super(PixelContrastLoss, self).__init__()
        self.configer = configer
        self.temperature = self.configer.get("contrast", "temperature")
        self.base_temperature = self.configer.get("contrast", "base_temperature")
        self.ignore_label = -1
        if self.configer.exists(
            "loss", "params"
        ) and "ce_ignore_index" in self.configer.get("loss", "params"):
            self.ignore_label = self.configer.get("loss", "params")["ce_ignore_index"]
        self.max_samples = self.configer.get("contrast", "max_samples")
        self.max_views = self.configer.get("contrast", "max_views")
        self.class_flags = self.configer.get("contrast", "classes")

    def _high_entropy_anchor_sampling(self, feats, label, logits, mask=None):
        """
        params:
        feats: pixel embedding features,dim * (B * H * W)
        label: gt of per pixel,(B * H * W)
        logits:output of net
        """
        batch_size, feat_dim = (feats.shape[0], feats.shape[-1])
        classes = []
        total_classes = 0
        for ii in range(batch_size):
            this_y = label[ii]
            this_classes = torch.unique(this_y)
            this_classes = [x for x in this_classes if x != self.ignore_label]
            this_classes = [
                x for x in this_classes if torch.sum(this_y == x) > self.max_views
            ]
            classes.append(this_classes)
            total_classes += len(this_classes)
        if total_classes == 0:
            return (None, None)
        n_view = self.max_samples // total_classes
        n_view = min(n_view, self.max_views)
        X_ = torch.zeros((total_classes, n_view, feat_dim), dtype=torch.float).cuda()
        y_ = torch.zeros(total_classes, dtype=torch.float).cuda()
        X_ptr = 0
        for ii in range(batch_size):
            this_y_hat = logits[ii]
            this_y = label[ii]
            this_classes = classes[ii]
            this_feats = feats[ii]
            for cls_id in this_classes:
                p = this_y_hat[cls_id, :]
                mask = this_y == cls_id
                plogp = -1.0 * p * torch.log(p)
                plogp = plogp[mask]
                mask_feats = this_feats[mask]
                num_hard = int(n_view / 2)
                num_easy = n_view - num_hard
                _, hard_indices = torch.topk(plogp, num_hard, dim=0)
                _, easy_indices = torch.topk(plogp, num_easy, largest=False, dim=0)
                indices = torch.cat((hard_indices, easy_indices), dim=0)
                X_[X_ptr, :, :] = mask_feats[indices, :].squeeze(1)
                y_[X_ptr] = cls_id
                X_ptr += 1
        return (X_, y_)

    def _build_anchor_sampling(self, X, y_hat, y):
        if len(y_hat.shape) == 3:
            y_hat = torch.argmax(y_hat, dim=1)
        if y.shape[2:] != y_hat.shape[2:]:
            b, h, w = y_hat.shape
            y = F.interpolate(y, (h, w), mode="nearest")
        batch_size = X.shape[0]
        classes = []
        total_classes = 0
        for ii in range(batch_size):
            this_y = y[ii]
            this_classes = torch.unique(this_y)
            this_classes = [x for x in this_classes if x != self.ignore_label]
            this_classes = [
                x for x in this_classes if torch.sum(this_y == x) > self.max_views
            ]
            classes.append(this_classes)
            total_classes += len(this_classes)
        n_view = self.max_samples // total_classes
        n_view = min(n_view, self.max_views)
        x_anchors = []
        y_anchors = []
        for ii in range(batch_size):
            this_y_hat = y_hat[ii]
            this_y = y[ii]
            this_classes = classes[ii]
            for cls_id in this_classes:
                if cls_id in self.class_flags:
                    hard_indices = (
                        (this_y == cls_id) & (this_y_hat != cls_id)
                    ).nonzero(as_tuple=False)
                    easy_indices = (
                        (this_y == cls_id) & (this_y_hat == cls_id)
                    ).nonzero(as_tuple=False)
                else:
                    easy_indices = (
                        (this_y == cls_id) & (this_y_hat == cls_id)
                    ).nonzero(as_tuple=False)
                    hard_indices = easy_indices
                num_hard = hard_indices.shape[0]
                num_easy = easy_indices.shape[0]
                if num_hard >= n_view / 2 and num_easy >= n_view / 2:
                    num_hard_keep = n_view // 2
                    num_easy_keep = n_view - num_hard_keep
                elif num_hard >= n_view / 2:
                    num_easy_keep = num_easy
                    num_hard_keep = n_view - num_easy_keep
                elif num_easy >= n_view / 2:
                    num_hard_keep = num_hard
                    num_easy_keep = n_view - num_hard_keep
                else:
                    Log.info(
                        "this shoud be never touched! {} {} {}".format(
                            num_hard, num_easy, n_view
                        )
                    )
                    continue
                perm = torch.randperm(num_hard)
                hard_indices = hard_indices[perm[:num_hard_keep]]
                perm = torch.randperm(num_easy)
                easy_indices = easy_indices[perm[:num_easy_keep]]
                indices = torch.cat((hard_indices, easy_indices), dim=0)
                x_anchors.append(X[ii, indices, :].squeeze(1))
                y_anchors.append(cls_id)
        X_ = torch.stack(x_anchors, dim=0)
        y_ = torch.stack(y_anchors, dim=0)
        return (X_, y_)

    def _hard_anchor_sampling(self, X, y_hat, y, mask=None):
        if len(y_hat.shape) == 3:
            y_hat = torch.argmax(y_hat, dim=1)
        if y.shape[2:] != y_hat.shape[2:]:
            b, h, w = y_hat.shape
            y = F.interpolate(y, (h, w), mode="nearest")
        batch_size, feat_dim = (X.shape[0], X.shape[-1])
        classes = []
        total_classes = 0
        for ii in range(batch_size):
            this_y = y_hat[ii]
            this_classes = torch.unique(this_y)
            this_classes = [x for x in this_classes if x != self.ignore_label]
            this_classes = [
                x for x in this_classes if torch.sum(this_y == x) > self.max_views
            ]
            classes.append(this_classes)
            total_classes += len(this_classes)
        if total_classes == 0:
            return (None, None)
        n_view = self.max_samples // total_classes
        n_view = min(n_view, self.max_views)
        X_ = torch.zeros((total_classes, n_view, feat_dim), dtype=torch.float).cuda()
        y_ = torch.zeros(total_classes, dtype=torch.float).cuda()
        X_ptr = 0
        for ii in range(batch_size):
            this_y_hat = y_hat[ii]
            this_y = y[ii]
            this_classes = classes[ii]
            for cls_id in this_classes:
                hard_indices = ((this_y_hat == cls_id) & (this_y != cls_id)).nonzero(
                    as_tuple=False
                )
                easy_indices = ((this_y_hat == cls_id) & (this_y == cls_id)).nonzero(
                    as_tuple=False
                )
                num_hard = hard_indices.shape[0]
                num_easy = easy_indices.shape[0]
                if num_hard >= n_view / 2 and num_easy >= n_view / 2:
                    num_hard_keep = n_view // 2
                    num_easy_keep = n_view - num_hard_keep
                elif num_hard >= n_view / 2:
                    num_easy_keep = num_easy
                    num_hard_keep = n_view - num_easy_keep
                elif num_easy >= n_view / 2:
                    num_hard_keep = num_hard
                    num_easy_keep = n_view - num_hard_keep
                else:
                    Log.info(
                        "this shoud be never touched! {} {} {}".format(
                            num_hard, num_easy, n_view
                        )
                    )
                    raise Exception
                perm = torch.randperm(num_hard)
                hard_indices = hard_indices[perm[:num_hard_keep]]
                perm = torch.randperm(num_easy)
                easy_indices = easy_indices[perm[:num_easy_keep]]
                indices = torch.cat((hard_indices, easy_indices), dim=0)
                X_[X_ptr, :, :] = X[ii, indices, :].squeeze(1)
                y_[X_ptr] = cls_id
                X_ptr += 1
        return (X_, y_)

    def _sample_negative(self, Q):
        class_num, cache_size, feat_size = Q.shape
        X_ = torch.zeros((class_num * cache_size, feat_size)).float().cuda()
        y_ = torch.zeros((class_num * cache_size, 1)).float().cuda()
        sample_ptr = 0
        for ii in range(class_num):
            this_q = Q[ii, :cache_size, :]
            X_[sample_ptr : sample_ptr + cache_size, ...] = this_q
            y_[sample_ptr : sample_ptr + cache_size, ...] = ii
            sample_ptr += cache_size
        return (X_, y_)

    def _contrastive(self, X_anchor, y_anchor, queue=None):
        anchor_num, n_view = (X_anchor.shape[0], X_anchor.shape[1])
        y_anchor = y_anchor.contiguous().view(-1, 1)
        anchor_count = n_view
        anchor_feature = torch.cat(torch.unbind(X_anchor, dim=1), dim=0)
        if queue is not None:
            X_contrast, y_contrast = self._sample_negative(queue)
            y_contrast = y_contrast.contiguous().view(-1, 1)
            contrast_count = 1
            contrast_feature = X_contrast
        else:
            y_contrast = y_anchor
            contrast_count = n_view
            contrast_feature = torch.cat(torch.unbind(X_anchor, dim=1), dim=0)
        mask = torch.eq(y_anchor, y_contrast.T).float().cuda()
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature
        )
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        mask = mask.repeat(anchor_count, contrast_count)
        neg_mask = 1 - mask
        logits_mask = torch.ones_like(mask).scatter_(
            1, torch.arange(anchor_num * anchor_count).view(-1, 1).cuda(), 0
        )
        mask = mask * logits_mask
        neg_logits = torch.exp(logits) * neg_mask
        neg_logits = neg_logits.sum(1, keepdim=True)
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits + neg_logits)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = -(self.base_temperature / self.temperature) * mean_log_prob_pos
        loss = loss.mean()
        return loss

    def _contrastive_new(self, X_anchor, y_anchor, queue=None):
        y_anchor = y_anchor.contiguous().view(-1, 1)
        anchor_feature = torch.cat(torch.unbind(X_anchor, dim=1), dim=0)
        if queue is not None:
            X_contrast, y_contrast = self._sample_negative(queue)
            y_contrast = y_contrast.contiguous().view(-1, 1)
            contrast_feature = X_contrast
        else:
            y_contrast = y_anchor
            contrast_feature = torch.cat(torch.unbind(X_anchor, dim=1), dim=0)
        mask = torch.eq(y_anchor, y_contrast.T).float().cuda()
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature
        )
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        neg_mask = 1 - mask
        exp_logits = torch.exp(logits)
        pos_feats = exp_logits[mask.bool()]
        neg_feats = exp_logits[neg_mask.bool()]
        loss = -1.0 * torch.mean(
            torch.log(pos_feats / (pos_feats.sum() + neg_feats.sum() + 1e-06) + 1e-06)
        )
        return loss

    def forward(self, feats, labels=None, predict=None, queue=None, mask=None):
        labels = labels.unsqueeze(1).float().clone()
        labels = torch.nn.functional.interpolate(
            labels, (feats.shape[2], feats.shape[3]), mode="nearest"
        )
        preds = torch.nn.functional.interpolate(
            predict,
            (feats.shape[2], feats.shape[3]),
            mode="bilinear",
            align_corners=True,
        )
        b, c, _, _ = preds.shape
        preds = torch.softmax(preds, dim=1)
        labels = labels.squeeze(1).long()
        assert labels.shape[-1] == feats.shape[-1], "{} {}".format(
            labels.shape, feats.shape
        )
        labels = labels.contiguous().view(b, -1)
        preds = preds.contiguous().view(b, c, -1)
        feats = feats.permute(0, 2, 3, 1)
        feats = feats.contiguous().view(feats.shape[0], -1, feats.shape[-1])
        feats_, labels_ = self._build_anchor_sampling(feats, preds, labels)
        if feats_ is None or labels_ is None:
            return 0.0
        loss = self._contrastive(feats_, labels_, queue=queue)
        return loss


class ContrastChangeLoss(nn.Module):

    def __init__(self, configer=None):
        super(ContrastChangeLoss, self).__init__()
        self.configer = configer
        ignore_index = -1
        if self.configer.exists(
            "loss", "params"
        ) and "ce_ignore_index" in self.configer.get("loss", "params"):
            ignore_index = self.configer.get("loss", "params")["ce_ignore_index"]
        Log.info("ignore_index: {}".format(ignore_index))
        self.loss_weight = self.configer.get("contrast", "loss_weight")
        key = self.configer.get("loss", "loss_type")
        self.seg_criterion = LOSS_DICT[key](configer=configer)
        self.contrast_criterion = PixelContrastLoss(configer=configer)
        self.with_memory = self.configer.exists("contrast", "with_memory")
        if self.with_memory:
            self.memory_size = self.configer.get("contrast", "memory_size")
            self.pixel_update_freq = self.configer.get("contrast", "pixel_update_freq")
        key = self.configer.get("loss", "loss_type")
        self.contrast_criterion = PixelContrastLoss(configer=configer)
        self.register_buffer("center_patch", torch.zeros(1, 1, 8192))
        self.t_algin = 0.05
        self.center_momentum_patch = 0.995

    def _dequeue_and_enqueue(
        self,
        keys,
        labels,
        segment_queue,
        segment_queue_ptr,
        pixel_queue,
        pixel_queue_ptr,
    ):
        batch_size = keys.shape[0]
        feat_dim = keys.shape[1]
        labels = F.interpolate(
            labels.unsqueeze(1).float(), size=keys.shape[2:], mode="nearest"
        ).long()
        for bs in range(batch_size):
            this_feat = keys[bs].contiguous().view(feat_dim, -1)
            this_label = labels[bs].contiguous().view(-1)
            this_label_ids = torch.unique(this_label)
            this_label_ids = [x for x in this_label_ids if x >= 0]
            for lb in this_label_ids:
                idxs = (this_label == lb).nonzero(as_tuple=False)
                feat = torch.mean(this_feat[:, idxs], dim=1).squeeze(1)
                ptr = int(segment_queue_ptr[lb] + 1) % self.memory_size
                segment_queue[lb, ptr, :] = nn.functional.normalize(
                    feat.view(-1), p=2, dim=0
                )
                segment_queue_ptr[lb] = (segment_queue_ptr[lb] + 1) % self.memory_size
                num_pixel = idxs.shape[0]
                perm = torch.randperm(num_pixel)
                K = min(num_pixel, self.pixel_update_freq)
                feat = this_feat[:, perm[:K]]
                feat = torch.transpose(feat, 0, 1)
                ptr = int(pixel_queue_ptr[lb])
                if ptr + K >= self.memory_size:
                    pixel_queue[lb, -K:, :] = nn.functional.normalize(feat, p=2, dim=1)
                    pixel_queue_ptr[lb] = 0
                else:
                    pixel_queue[lb, ptr : ptr + K, :] = nn.functional.normalize(
                        feat, p=2, dim=1
                    )
                    pixel_queue_ptr[lb] = (pixel_queue_ptr[lb] + 1) % self.memory_size

    def patch_algin_loss(self, feats_label, feats_img, mask):
        """Masked KL(student map || frozen image teacher), averaged per image."""
        mask = 1.0 - mask
        # Float32 log probabilities avoid underflow under autocast. PyTorch's
        # target is the student here, so gradients must flow through log_target.
        student_log = F.log_softmax(feats_label.float(), dim=-1)
        teacher_log = F.log_softmax(feats_img[1].detach().float(), dim=-1)
        loss = F.kl_div(
            teacher_log, student_log, reduction="none", log_target=True
        ).sum(dim=-1)
        mask = mask.flatten(-2, -1)
        loss = torch.sum(loss * mask.float(), dim=-1) / mask.sum(dim=-1).clamp(min=1.0)
        return loss.mean()

    def forward(
        self,
        model,
        feats_left,
        feats_right,
        feats_head_right,
        target,
        masks,
        with_embed=False,
        patch_algin=True,
        pixel_algin=True,
        ce_loss=True,
    ):
        h, w = (target.size(1), target.size(2))
        if self.training:
            preds = model(
                feats_left,
                feats_right,
                targets=target,
                mask=masks,
                with_embed=with_embed,
            )
        else:
            preds = model(feats_left, feats_right, mask=masks, with_embed=False)
        assert "pred" in preds
        assert "embed" in preds
        assert "embed_algin" in preds
        feats_shp = preds["embed_algin"]
        pred = preds["pred"]
        embedding = preds["embed"]
        if hasattr(model, "module"):
            if hasattr(model.module, "segment_queue") and hasattr(
                model.module, "pixel_queue"
            ):
                preds["segment_queue"] = model.module.segment_queue
                preds["pixel_queue"] = model.module.pixel_queue
        elif hasattr(model, "segment_queue") and hasattr(model, "pixel_queue"):
            preds["segment_queue"] = model.segment_queue
            preds["pixel_queue"] = model.pixel_queue
        if "segment_queue" in preds:
            segment_queue = preds["segment_queue"]
        else:
            segment_queue = None
        if "pixel_queue" in preds:
            pixel_queue = preds["pixel_queue"]
        else:
            pixel_queue = None
        pred = F.interpolate(
            input=pred, size=(h, w), mode="bilinear", align_corners=True
        )
        loss_l = self.seg_criterion(pred, target)
        if isinstance(loss_l, dict):
            loss_l = loss_l["loss"]
        if not ce_loss:
            loss_l = 0.0 * loss_l
        if patch_algin:
            loss_patch_algin = self.patch_algin_loss(feats_shp, feats_head_right, masks)
        else:
            loss_patch_algin = 0.0 * loss_l
        if (
            segment_queue is not None
            and pixel_queue is not None
            and with_embed
            and pixel_algin
        ):
            queue = torch.cat((segment_queue, pixel_queue), dim=1)
            embedding = F.interpolate(
                embedding,
                (target.shape[1], target.shape[2]),
                mode="bilinear",
                align_corners=True,
            )
            target_down = (
                F.interpolate(
                    target.unsqueeze(0).float(),
                    (embedding.shape[2], embedding.shape[3]),
                    mode="nearest",
                )
                .squeeze(0)
                .long()
            )
            loss_pixel_algin = 0.1 * self.contrast_criterion(
                embedding, target_down, pred, queue
            )
        else:
            loss_pixel_algin = 0 * loss_l
        if (
            segment_queue is not None
            and pixel_queue is not None
            and with_embed
            and (embedding is not None)
            and pixel_algin
        ):
            keys = preds["key"]
            labels = preds["lb_key"]
            if hasattr(model, "module"):
                self._dequeue_and_enqueue(
                    keys,
                    labels,
                    model.module.segment_queue,
                    model.module.segment_queue_ptr,
                    model.module.pixel_queue,
                    model.module.pixel_queue_ptr,
                )
            else:
                self._dequeue_and_enqueue(
                    keys,
                    labels,
                    model.segment_queue,
                    model.segment_queue_ptr,
                    model.pixel_queue,
                    model.pixel_queue_ptr,
                )
        loss_total = loss_l + self.loss_weight * (loss_patch_algin + loss_pixel_algin)
        return {
            "preds": preds,
            "loss": loss_total,
            "loss_l": loss_l,
            "loss_patch_algin": loss_patch_algin,
            "loss_pixel_algin": loss_pixel_algin,
        }
