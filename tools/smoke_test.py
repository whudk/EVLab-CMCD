"""Exercise the actual CMPANet trainer with synthetic images and a real teacher."""

import argparse
import json
import math
import random
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained", required=True)
    parser.add_argument("--configs", default=str(ROOT / "configs/cmpanet_small_head24.json"))
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--fp16", choices=["true", "false"], default="true")
    parser.add_argument("--steps", type=int, default=8)
    args = parser.parse_args()
    import cv2
    import numpy as np
    import torch
    from main import build_parser, load_config, validate_config
    from lib.utils.tools.configer import Configer
    from segmentor.trainer_SSL import Trainer_SSL_CHG

    if not torch.cuda.is_available():
        raise RuntimeError("This test requires a CUDA GPU.")
    torch.set_num_threads(4)
    torch.manual_seed(304)
    random.seed(304)
    np.random.seed(304)
    rng = np.random.default_rng(304)
    with tempfile.TemporaryDirectory(prefix="cmpanet_smoke_") as directory:
        tmp = Path(directory)
        lines = []
        for i in range(4):
            left = rng.integers(2, 255, (args.size, args.size, 3), dtype=np.uint8)
            right = rng.integers(2, 255, left.shape, dtype=np.uint8)
            label = np.zeros(left.shape[:2], dtype=np.uint8)
            label[args.size // 4 : args.size // 2, args.size // 4 : args.size // 2] = (
                255
            )
            paths = [tmp / f"{i}_{name}.png" for name in ("left", "right", "label")]
            for path, array in zip(paths, (left, right, label)):
                cv2.imencode(".png", array)[1].tofile(path)
            lines.append(" ".join(map(str, paths)))
        train = tmp / "train.txt"
        val = tmp / "val.txt"
        train.write_text("\n".join(lines), encoding="utf-8")
        val.write_text(lines[0], encoding="utf-8")
        cli = build_parser().parse_args(
            [
                "--configs", args.configs,
                "--suptxt",
                str(train),
                "--valtxt",
                str(val),
                "--pretrained",
                str(Path(args.pretrained).resolve()),
                "--train_batch_size",
                "2",
                "--val_batch_size",
                "1",
                "--epochs",
                str(math.ceil(args.steps / 2) + 1),
                "--max_iters",
                str(args.steps),
                "--test_interval",
                "1",
                "--fp16",
                args.fp16,
                "--checkpoints_root",
                str(tmp / "checkpoints"),
            ]
        )
        cfg = load_config(cli)
        cfg["logging"]["tb_logger"] = None
        cfg["solver"]["display_iter"] = 1
        cfg["train"]["data_transformer"]["input_size"] = [args.size, args.size]
        cfg["val"]["data_transformer"]["input_size"] = [args.size, args.size]
        cfg["train_trans"]["resize"]["target_size"] = [args.size, args.size]
        cfg["train_trans"]["random_crop"]["crop_size"] = [args.size, args.size]
        cfg["val_trans"]["resize"]["min_side_length"] = args.size
        # Exercise the post-warmup projection branch as well.
        cfg["contrast"]["warmup_iters"] = 1
        validate_config(cfg)
        trainer = Trainer_SSL_CHG(Configer(config_dict=cfg))
        assert len(trainer.train_loader) == 2
        assert len(trainer.val_loader) == 1
        before = trainer.decoder.patch_head[-1].weight.detach().clone()
        observed_modes = []
        losses = []
        updates = []
        step_hook = trainer.optimizer.register_step_post_hook(
            lambda *unused: updates.append(True)
        )

        def observe(module, inputs, output):
            assert inputs[1].shape[-2:] == (args.size, args.size)
            if module.training:
                observed_modes.append(trainer.decoder.training)
                losses.append(float(output["loss"].detach()))
                assert torch.isfinite(output["loss"]).all()

        hook = trainer.loss.register_forward_hook(observe)
        # Avoid writing several full-size checkpoints during a smoke run.
        saves = []
        save_net = trainer.module_runner.save_net
        trainer.module_runner.save_net = (
            lambda net, save_mode="iters", **kw: saves.append(save_mode)
        )
        trainer.train()
        hook.remove()
        step_hook.remove()
        assert cfg["iters"] == args.steps, cfg["iters"]
        assert observed_modes == [True] * args.steps, observed_modes
        assert len(losses) == args.steps and all(map(math.isfinite, losses))
        assert (
            updates
        ), f"No optimizer updates; final AMP scale={trainer.scaler.get_scale()}"
        assert not torch.equal(before, trainer.decoder.patch_head[-1].weight)
        assert all(
            p.grad is None and not p.requires_grad for p in trainer.encoder.parameters()
        )
        assert math.isfinite(float(cfg["val_loss"]))
        assert "performance" in saves and "val_loss" in saves
        # Verify actual checkpoint serialization and loading with a small module.
        small = torch.nn.Linear(3, 2).cuda()
        save_net(small, save_mode="performance")
        name = cfg["checkpoints"]["checkpoints_name"]
        checkpoint = tmp / "checkpoints" / name / (name + "_latest.pth")
        restored = trainer.module_runner.load_net(
            torch.nn.Linear(3, 2), str(checkpoint)
        )
        assert torch.equal(small.weight, restored.weight)
        # Object counting must skip background and include the last component.
        target = torch.zeros((1, 8, 8), dtype=torch.long)
        target[:, 2:4, 2:4] = 1
        assert trainer.evaluator.evaluation_object(target, target) == (1, 1, 1, 1)
        print(
            json.dumps(
                dict(
                    status="PASS",
                    steps=cfg["iters"],
                    optimizer_updates=len(updates),
                    final_amp_scale=trainer.scaler.get_scale(),
                    size=args.size,
                    fp16=args.fp16,
                    train_losses=losses,
                    final_val_loss=float(cfg["val_loss"]),
                    checkpoint_roundtrip=True,
                    teacher_frozen=True,
                    validation_mode_restored=True,
                    peak_gpu_mb=round(torch.cuda.max_memory_allocated() / 1024**2),
                ),
                indent=2,
            )
        )


if __name__ == "__main__":
    run()
