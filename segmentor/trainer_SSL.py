from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from lib.utils.tools.average_meter import AverageMeter
from lib.datasets.data_loader import DataLoader
from lib.loss.loss_manager import LossManager
from lib.models.model_manager import ModelManager
from lib.utils.tools.logger import Logger as Log
from segmentor.tools.module_runner import ModuleRunner
from segmentor.tools.optim_scheduler import OptimScheduler
from segmentor.tools.evaluator import get_evaluator
from segmentor.tools.get_tb_logger import get_tb_logger
import torch.nn as nn
from lib.utils.distributed import get_world_size, get_rank, is_distributed
import torch
import time
from lib.models.backbones.head import iBOTHead
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
import torch.distributed as dist
from lib.models.backbones.head import MultiCropWrapper


class Trainer_SSL_SEG(object):

    def __init__(self, configer):
        self.configer = configer
        self.batch_time = AverageMeter()
        self.foward_time = AverageMeter()
        self.backward_time = AverageMeter()
        self.loss_time = AverageMeter()
        self.data_time = AverageMeter()
        self.train_losses = AverageMeter()
        self.val_losses = AverageMeter()
        self.loss_manager = LossManager(configer)
        self.module_runner = ModuleRunner(configer)
        self.model_manager = ModelManager(configer)
        self.data_loader = DataLoader(configer)
        self.optim_scheduler = OptimScheduler(configer)
        self.evaluator = get_evaluator(configer, self)
        self.tb_logger = get_tb_logger(configer)
        self.teacher = None
        self.student = None
        self.train_loader = None
        self.optimizer = None
        self.scheduler = None
        self.running_score = None
        self.__init_trainer()

    def __init_trainer(self):
        self.encoder = self.model_manager.build_encoder(
            encoder_name=self.configer.get("network", "image_encoder")["name"],
            net_params=self.configer.get("network", "image_encoder")["params"],
        )
        pretrained = self.configer.get("network", "image_encoder")["pretrained"]
        if self.configer.exists("network", "image_encoder"):
            head_params = self.configer.get("network", "image_encoder")["head"]
            self.encoder = MultiCropWrapper(
                self.encoder,
                iBOTHead(
                    768,
                    out_dim=head_params["out_dim"],
                    patch_out_dim=head_params["patch_out_dim"],
                    norm=head_params["norm_in_head"],
                    act=head_params["act_in_head"],
                    norm_last_layer=head_params["norm_last_layer"],
                    shared_head=head_params["shared_head_student"],
                ),
            ).cuda()
            self.module_runner.restart_from_checkpoint(
                ckp_path=pretrained, teacher=self.encoder
            )
        self.module_runner.to_device(self.encoder)
        self.module_runner._make_parallel(self.encoder)
        self.decoder = self.model_manager.build_decoder()
        self.decoder = self.module_runner.load_net(self.decoder)
        if not dist.is_initialized():
            if hasattr(self.decoder, "module"):
                self.decoder = self.decoder.module
        self.use_fp16 = self.configer.get("train", "fp16")
        self.scaler = GradScaler(enabled=self.use_fp16)
        Log.info("use fp16={} for trainning".format(self.configer.get("train", "fp16")))
        Log.info("self.teacher.get_semi_trainloader()")
        self.train_loader, self.unsup_loader, self.val_loader = (
            self.data_loader.get_semi_trainloader()
        )
        Log.info(
            "Params Group Method: {}".format(self.configer.get("optim", "group_method"))
        )
        if self.configer.get("optim", "group_method") == "decay":
            params_group = self.group_weight(self.decoder)
        else:
            assert self.configer.get("optim", "group_method") is None
            params_group = self._get_parameters(self.decoder)
        self.optimizer, _ = self.optim_scheduler.init_optimizer(params_group)
        self.scheduler = self.optim_scheduler.cosine_scheduler(
            base_value=self.configer.get("lr", "base_lr"),
            final_value=1e-06,
            epochs=self.configer.get("train", "epoch"),
            niter_per_ep=len(self.train_loader),
            warmup_epochs=self.configer.get("train", "warm_up_epoch"),
        )
        self.with_contrast = True if self.configer.exists("contrast") else False
        if self.configer.exists("contrast", "warmup_iters"):
            self.contrast_warmup_iters = self.configer.get("contrast", "warmup_iters")
        else:
            self.contrast_warmup_iters = 0
        if self.with_contrast:
            self.loss = self.loss_manager.build_loss(
                self.configer.get("contrast", "loss"), model=self.decoder
            )
        else:
            self.loss = self.loss_manager.build_loss()
        self.loss = self.module_runner.to_device(self.loss)
        freeze_encoder = self.configer.get("network", "image_encoder")
        if freeze_encoder:
            if hasattr(self.encoder, "module"):
                for p in self.encoder.module.parameters():
                    p.requires_grad = False
            else:
                for p in self.encoder.parameters():
                    p.requires_grad = False
        param_m = sum((p.numel() for name, p in self.decoder.named_parameters()))
        param_m = param_m / 1000000.0
        Log.info(f"Params : {param_m:.2f}M parameters")

    @staticmethod
    def group_weight(module):
        group_decay = []
        group_no_decay = []
        for m in module.modules():
            if isinstance(m, nn.Linear):
                group_decay.append(m.weight)
                if m.bias is not None:
                    group_no_decay.append(m.bias)
            elif isinstance(m, nn.modules.conv._ConvNd):
                group_decay.append(m.weight)
                if m.bias is not None:
                    group_no_decay.append(m.bias)
            else:
                if hasattr(m, "weight"):
                    group_no_decay.append(m.weight)
                if hasattr(m, "bias"):
                    group_no_decay.append(m.bias)
        assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)
        groups = [
            dict(params=group_decay),
            dict(params=group_no_decay, weight_decay=0.0),
        ]
        return groups

    def _get_parameters(self, model):
        bb_lr = []
        nbb_lr = []
        params_dict = dict(model.named_parameters())
        for key, value in params_dict.items():
            if "backbone" not in key:
                nbb_lr.append(value)
            else:
                bb_lr.append(value)
        params = [
            {"params": bb_lr, "lr": self.configer.get("lr", "base_lr")},
            {
                "params": nbb_lr,
                "lr": self.configer.get("lr", "base_lr")
                * self.configer.get("lr", "nbb_mult"),
            },
        ]
        return params


class Trainer_SSL_CHG(Trainer_SSL_SEG):

    def __init__(self, configer=None):
        super().__init__(configer=configer)
        self.mode = self.configer.get("train", "mode")
        if self.configer.exists("loss", "patch_algin"):
            self.patch_algin = self.configer.get("loss", "patch_algin")
            self.patch_algin_loss = AverageMeter()
        if self.configer.exists("loss", "pixel_algin"):
            self.pixel_algin = self.configer.get("loss", "pixel_algin")
            self.pixel_algin_loss = AverageMeter()
        if self.configer.exists("loss", "ce_loss"):
            self.ce_loss = self.configer.get("loss", "ce_loss")

    def __valshpcd(self, data_loader=None):
        sub_dir = self.configer.get("checkpoints", "checkpoints_name")
        if self.patch_algin:
            sub_dir = sub_dir + "_patch_algin"
        if self.pixel_algin:
            sub_dir = sub_dir + "_pixel_algin"
        if isinstance(self.decoder, nn.Module):
            self.decoder.eval()
        self.loss.eval()
        self.decoder.eval()
        self.encoder.eval()
        start_time = time.time()
        data_loader = self.val_loader if data_loader is None else data_loader
        gts_n, preds_n, chks_n, tures_n = (0, 0, 0, 0)
        for j, (inputs, targets, masks, img_metas) in enumerate(data_loader):
            if j % 10 == 0:
                Log.info("{} images processed\n".format(j))
            masks = masks.cuda()
            inputs = inputs.cuda()
            targets = targets.cuda()
            inputs_left, inputs_right = torch.split(inputs, 3, dim=1)
            with torch.no_grad():
                feats = self.encoder.backbone.get_feats(inputs_right)
                feats_head = self.encoder.head(feats[-1])
                feats = [feat[:, 1:, :] for feat in feats]
                rets = self.loss(
                    self.decoder,
                    inputs_left,
                    feats,
                    feats_head,
                    target=targets.detach(),
                    masks=masks,
                    with_embed=False,
                    patch_algin=self.patch_algin,
                    pixel_algin=self.pixel_algin,
                    ce_loss=self.ce_loss,
                )
                loss = rets["loss"]
                outputs_chg = rets["preds"]["pred"]
                outputs_chg = F.interpolate(
                    outputs_chg,
                    (targets.shape[1], targets.shape[2]),
                    mode="bilinear",
                    align_corners=True,
                )
                outputs_chg = torch.argmax(outputs_chg, dim=1)
                batch_size = inputs_left.shape[0]
                if get_rank() == 0 or not is_distributed():
                    pass
                if not is_distributed() or get_rank() == 0:
                    self.val_losses.update(loss.item(), batch_size)
                    self.evaluator.update_score_seg(outputs_chg, targets)
                    pred_n, gt_n, chk_n, ture_n = self.evaluator.evaluation_object(
                        outputs_chg, targets
                    )
                    preds_n = preds_n + pred_n
                    gts_n = gts_n + gt_n
                    chks_n = chks_n + chk_n
                    tures_n = tures_n + ture_n
            self.batch_time.update(time.time() - start_time)
            start_time = time.time()
        if not is_distributed() or get_rank() == 0:
            self.evaluator.update_performance()
            self.evaluator.update_acc()
        self.configer.update(["val_loss"], self.val_losses.avg)
        if self.configer.get("network", "resume_val") is not True:
            if isinstance(self.decoder, nn.Module):
                self.module_runner.save_net(self.decoder, save_mode="performance")
                self.module_runner.save_net(self.decoder, save_mode="val_loss")
            else:
                ema_to_save = (
                    self.decoder.ema.module
                    if hasattr(self.decoder.ema, "module")
                    else self.decoder.ema
                )
                self.module_runner.save_net(ema_to_save, save_mode="performance")
                self.module_runner.save_net(ema_to_save, save_mode="val_loss")
        if not is_distributed() or get_rank() == 0:
            Log.info(
                "Test Time {batch_time.sum:.3f}s, ({batch_time.avg:.3f})\tLoss {loss.avg:.8f}\n".format(
                    batch_time=self.batch_time, loss=self.val_losses
                )
            )
            self.evaluator.print_scores()
            recall = chks_n / max(gts_n, 1)
            precision = tures_n / max(preds_n, 1)
            f1_score = (
                2 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0
            )
            Log.info(
                "evaluation by object\n F1 Score:{} Precision:{} Recall:{}".format(
                    f1_score, precision, recall
                )
            )
        self.batch_time.reset()
        self.val_losses.reset()
        self.evaluator.reset()

    def __train_shpcd(self):
        self.encoder.eval()
        self.decoder.train()
        self.loss.train()
        if self.use_fp16:
            scaler = self.scaler
        if hasattr(self.train_loader.sampler, "set_epoch"):
            self.train_loader.sampler.set_epoch(self.configer.get("epoch"))
        for j, (inputs_l, targets_l, masks, img_metas) in enumerate(self.train_loader):
            if self.configer.get("iters") >= self.configer.get("solver", "max_iters"):
                break
            self.module_runner.get_lr_from_scheduler(self.optimizer, self.scheduler)
            start_time = time.time()
            if not (targets_l == 1).any():
                continue
            inputs_l = inputs_l.cuda()
            inputs_left, inputs_right = torch.split(inputs_l, 3, dim=1)
            targets_l = targets_l.cuda()
            masks = masks.cuda()
            self.data_time.update(time.time() - start_time)
            foward_start_time = time.time()
            self.foward_time.update(time.time() - foward_start_time)
            loss_start_time = time.time()
            with torch.no_grad():
                if hasattr(self.encoder, "backbone"):
                    feats = self.encoder.backbone.get_feats(inputs_right)
                    feats_head = self.encoder.head(feats[-1])
                    feats = [feat[:, 1:, :] for feat in feats]
                else:
                    feats = self.encoder(inputs_right)
                    feats_head = feats[-1]
                    feats = feats[2:]
            with_embed = (
                True
                if self.configer.get("iters")
                >= self.configer.get("contrast", "warmup_iters")
                else False
            )
            if self.use_fp16:
                with autocast():
                    loss_dict = self.loss(
                        self.decoder,
                        inputs_left,
                        feats,
                        feats_head,
                        target=targets_l.detach(),
                        masks=masks,
                        with_embed=with_embed,
                        patch_algin=self.patch_algin,
                        pixel_algin=self.pixel_algin,
                        ce_loss=self.ce_loss,
                    )
            else:
                loss_dict = self.loss(
                    self.decoder,
                    inputs_left,
                    feats,
                    feats_head,
                    target=targets_l.detach(),
                    masks=masks,
                    with_embed=with_embed,
                    patch_algin=self.patch_algin,
                    pixel_algin=self.pixel_algin,
                    ce_loss=self.ce_loss,
                )
            backward_loss = display_loss = loss_dict["loss"]
            if is_distributed():
                import torch.distributed as dist

                def reduce_tensor(inp):
                    """
                    Reduce the loss from all processes so that
                    process with rank 0 has the averaged results.
                    """
                    world_size = get_world_size()
                    if world_size < 2:
                        return inp
                    with torch.no_grad():
                        reduced_inp = inp
                        dist.all_reduce(reduced_inp)
                    return reduced_inp

                display_loss = reduce_tensor(display_loss) / get_world_size()
            self.train_losses.update(display_loss.item(), inputs_left.shape[0])
            if self.pixel_algin:
                self.pixel_algin_loss.update(
                    loss_dict["loss_pixel_algin"].item(), inputs_left.shape[0]
                )
            if self.patch_algin:
                self.patch_algin_loss.update(
                    loss_dict["loss_patch_algin"].item(), inputs_left.shape[0]
                )
            self.loss_time.update(time.time() - loss_start_time)
            self.optimizer.zero_grad()
            backward_start_time = time.time()
            if self.use_fp16:
                backward_start_time = time.time()
                scaler.scale(backward_loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
            else:
                backward_loss.backward()
                self.optimizer.step()
            self.backward_time.update(time.time() - backward_start_time)
            self.batch_time.update(time.time() - start_time)
            start_time = time.time()
            self.configer.plus_one("iters")
            if self.configer.get("iters") % self.configer.get(
                "solver", "display_iter"
            ) == 0 and (not is_distributed() or get_rank() == 0):
                Log.info(
                    "Train Epoch: {0}\tTrain Iteration: {1}\tTime {batch_time.sum:.3f}s / {2}iters, ({batch_time.avg:.3f})\tForward Time {foward_time.sum:.3f}s / {2}iters, ({foward_time.avg:.3f})\tBackward Time {backward_time.sum:.3f}s / {2}iters, ({backward_time.avg:.3f})\tLoss Time {loss_time.sum:.3f}s / {2}iters, ({loss_time.avg:.3f})\tData load {data_time.sum:.3f}s / {2}iters, ({data_time.avg:3f})\nLearning rate = {3}\tLoss = {loss.val:.8f} (ave = {loss.avg:.8f})\tPixel_algin_loss = {pixel_algin_loss.val:.8f} (ave = {pixel_algin_loss.avg:.8f})\tPatch_algin_loss = {patch_algin_loss.val:.8f} (ave = {patch_algin_loss.avg:.8f})\n".format(
                        self.configer.get("epoch"),
                        self.configer.get("iters"),
                        self.configer.get("solver", "display_iter"),
                        self.module_runner.get_lr_from_scheduler(
                            self.optimizer, self.scheduler
                        ),
                        batch_time=self.batch_time,
                        foward_time=self.foward_time,
                        backward_time=self.backward_time,
                        loss_time=self.loss_time,
                        data_time=self.data_time,
                        loss=self.train_losses,
                        pixel_algin_loss=self.pixel_algin_loss,
                        patch_algin_loss=self.patch_algin_loss,
                    )
                )
                self.loss_time.reset()
                self.batch_time.reset()
                self.foward_time.reset()
                self.backward_time.reset()
                self.data_time.reset()
                self.train_losses.reset()
                self.patch_algin_loss.reset()
                self.pixel_algin_loss.reset()
            if self.tb_logger is not None:
                i_iter = self.configer.get("iters")
                self.tb_logger.add_scalar(
                    "lr",
                    self.module_runner.get_lr_from_scheduler(
                        self.optimizer, self.scheduler
                    ),
                    i_iter,
                )
                self.tb_logger.add_scalars("loss", {"total_loss": display_loss}, i_iter)
            torch.cuda.empty_cache()
            if (
                self.configer.get("iters")
                % self.configer.get("solver", "test_interval")
                == 0
            ):
                self.__valshpcd()
                self.decoder.train()
                self.loss.train()
        self.configer.plus_one("epoch")

    def train(self):
        if self.configer.get("network", "resume_val"):
            self.__valshpcd()
            return
        while self.configer.get("epoch") < self.configer.get(
            "train", "epoch"
        ) and self.configer.get("iters") < self.configer.get("solver", "max_iters"):
            self.__train_shpcd()
        self.__valshpcd()
        if self.tb_logger is not None:
            self.tb_logger.close()
