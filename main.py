"""Single-GPU CMPANet training entry point."""

import argparse
import json
import os
from pathlib import Path
import random
import time

ROOT = Path(__file__).resolve().parent


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if value.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Expected true/false or 1/0.")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument(
        "--configs", default=str(ROOT / "configs/shpcd/configer_shpcd_vit_base.json")
    )
    p.add_argument("--phase", default="train_ssl_cd", choices=["train_ssl_cd"])
    p.add_argument("--gpu", "--gpuid", dest="gpu", type=int, nargs="+", default=[0])
    p.add_argument(
        "--bn_type", dest="network:bn_type", choices=["torchbn"], default=None
    )
    p.add_argument("--suptxt", dest="data:suptxt", required=True)
    p.add_argument(
        "--unsuptxt",
        dest="data:unsuptxt",
        default=None,
        help="Compatibility argument; this training branch does not optimize on it.",
    )
    p.add_argument("--valtxt", dest="data:valtxt", required=True)
    p.add_argument(
        "--pretrained", help="iBOT checkpoint containing the teacher backbone and head."
    )
    p.add_argument("--base_lr", dest="lr:base_lr", type=float, default=None)
    p.add_argument(
        "--test_interval", dest="solver:test_interval", type=int, default=None
    )
    p.add_argument("--max_iters", dest="solver:max_iters", type=int, default=None)
    p.add_argument("--epochs", dest="train:epoch", type=int, default=None)
    p.add_argument(
        "--train_batch_size", dest="train:batch_size", type=int, default=None
    )
    p.add_argument("--val_batch_size", dest="val:batch_size", type=int, default=None)
    p.add_argument("--workers", dest="data:workers", type=int, default=None)
    p.add_argument("--fp16", dest="train:fp16", type=str2bool, default=None)
    p.add_argument(
        "--patch_algin",
        "--patch_align",
        dest="loss:patch_algin",
        type=str2bool,
        nargs="?",
        const=True,
        default=None,
    )
    p.add_argument(
        "--checkpoints_name", dest="checkpoints:checkpoints_name", default=None
    )
    p.add_argument(
        "--checkpoints_root", dest="checkpoints:checkpoints_root", default="checkpoints"
    )
    p.add_argument("--log_file", dest="logging:log_file", default=None)
    p.add_argument(
        "--resume",
        dest="network:resume",
        default=None,
        help="Load decoder weights (optimizer state is not restored).",
    )
    p.add_argument(
        "--resume_eval_val",
        dest="network:resume_val",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
    )
    p.add_argument("--seed", type=int, default=304)
    p.add_argument("--memory-fraction", type=float, default=0.60,
                   help="CUDA allocator limit as a fraction of device memory.")
    p.add_argument(
        "--check-config",
        action="store_true",
        help="Validate configuration and input files without importing PyTorch.",
    )
    return p


def load_config(args):
    with Path(args.configs).open(encoding="utf-8") as handle:
        cfg = json.load(handle)
    for key, value in vars(args).items():
        if ":" in key and value is not None:
            section, name = key.split(":")
            cfg.setdefault(section, {})[name] = value
    if args.pretrained:
        cfg["network"]["image_encoder"]["pretrained"] = args.pretrained
    cfg.update(phase=args.phase, gpu=args.gpu, seed=args.seed, project_dir=str(ROOT))
    cfg["data"].setdefault("unsuptxt", cfg["data"]["valtxt"])
    cfg["data"].setdefault("drop_last", False)
    cfg["optim"].setdefault("group_method", None)
    cfg["lr"].setdefault("nbb_mult", 1.0)
    cfg["lr"].setdefault("is_warm", False)
    for key, value in dict(
        resume=None,
        resume_strict=True,
        resume_continue=False,
        resume_val=False,
        gathered=True,
    ).items():
        cfg["network"].setdefault(key, value)
    cfg["checkpoints"].setdefault("save_epoch", 1)
    # Each named run gets its own checkpoint directory.
    cfg["checkpoints"]["checkpoints_dir"] = cfg["checkpoints"]["checkpoints_name"]
    return cfg


def validate_config(cfg):
    if len(cfg["gpu"]) != 1 or cfg["gpu"][0] < 0:
        raise ValueError("This clean extraction supports one GPU, e.g. --gpu 0.")
    if cfg["task"] != "shpcd" or cfg["train"]["mode"] != "shpcd":
        raise ValueError("Only the supplied shpcd training branch is included.")
    if cfg["network"]["bn_type"] != "torchbn":
        raise ValueError("Use --bn_type torchbn.")
    if cfg["loss"]["pixel_algin"]:
        raise ValueError(
            "This release is scoped to the supplied pixel_algin=false configuration."
        )
    for section, key in [
        ("train", "epoch"),
        ("train", "batch_size"),
        ("val", "batch_size"),
        ("solver", "max_iters"),
        ("solver", "test_interval"),
        ("solver", "display_iter"),
    ]:
        if cfg[section][key] <= 0:
            raise ValueError(f"{section}.{key} must be positive.")
    if cfg["train"]["batch_size"] < 2:
        raise ValueError(
            "UperHead uses BatchNorm after 1x1 pooling; training batch size must be at least 2."
        )
    if not 0 <= cfg["train"]["warm_up_epoch"] < cfg["train"]["epoch"]:
        raise ValueError("warm_up_epoch must be smaller than train.epoch.")
    if cfg["data"]["workers"] < 0 or cfg["lr"]["base_lr"] <= 0:
        raise ValueError("workers must be nonnegative and base_lr positive.")
    size = cfg["train"]["data_transformer"]["input_size"]
    if size[0] != size[1] or size[0] % 32:
        raise ValueError("Use square input sizes divisible by 32.")
    for key in ("suptxt", "unsuptxt", "valtxt"):
        path = Path(cfg["data"][key]).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"{key}: {path}")
        cfg["data"][key] = str(path)
        lines = [
            line
            for line in path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        ]
        if not lines:
            raise ValueError(f"Empty dataset list: {path}")
        for index, line in enumerate(lines, 1):
            fields = line.split()
            if len(fields) != 3:
                raise ValueError(
                    f"{path}:{index}: expected three whitespace-separated paths."
                )
            for item in fields:
                if not Path(item).is_file():
                    raise FileNotFoundError(f"{path}:{index}: {item}")
        if key == "suptxt" and len(lines) < 2:
            raise ValueError("Training requires at least two samples for BatchNorm.")
        # Avoid a singleton final batch, which cannot pass the PPM BatchNorm layer.
        if key == "suptxt" and len(lines) % cfg["train"]["batch_size"] == 1:
            cfg["data"]["drop_last"] = True
    weight = Path(cfg["network"]["image_encoder"]["pretrained"]).expanduser().resolve()
    if not weight.is_file():
        raise FileNotFoundError(
            f"Pretrained teacher checkpoint missing: {weight}. Set --pretrained PATH."
        )
    cfg["network"]["image_encoder"]["pretrained"] = str(weight)
    if cfg["network"]["resume"] and not Path(cfg["network"]["resume"]).is_file():
        raise FileNotFoundError(cfg["network"]["resume"])
    return cfg


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not 0 < args.memory_fraction <= 1:
        parser.error("--memory-fraction must be in (0, 1].")
    try:
        cfg = validate_config(load_config(args))
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    if args.check_config:
        print("Configuration, dataset lists, sample paths and checkpoint path: OK")
        print(
            "Unsupervised list is accepted for compatibility; it does not contribute training loss."
        )
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg["gpu"][0])
    import numpy as np
    import torch
    from lib.utils.tools.configer import Configer
    from lib.utils.tools.logger import Logger
    from segmentor.trainer_SSL import Trainer_SSL_CHG

    if not torch.cuda.is_available():
        parser.error(
            "CUDA is unavailable. Install a CUDA-enabled PyTorch build and check your driver."
        )
    random.seed(args.seed)
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    log_path = Path(cfg["logging"]["log_file"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cfg["logging"]["log_file"] = (
        str(log_path) + "_" + time.strftime("%Y-%m-%d_%H%M%S") + ".log"
    )
    Logger.init(
        logfile_level=cfg["logging"]["logfile_level"],
        stdout_level=cfg["logging"]["stdout_level"],
        log_file=cfg["logging"]["log_file"],
        log_format=cfg["logging"]["log_format"],
        rewrite=cfg["logging"]["rewrite"],
    )
    Logger.info("CMPANet: single-GPU shpcd; unsuptxt is not used for optimization.")
    Trainer_SSL_CHG(Configer(config_dict=cfg)).train()


if __name__ == "__main__":
    main()
