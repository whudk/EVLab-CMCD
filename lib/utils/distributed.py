import torch


def is_distributed():
    return torch.distributed.is_initialized()


def get_world_size():
    return torch.distributed.get_world_size() if is_distributed() else 1


def get_rank():
    return torch.distributed.get_rank() if is_distributed() else 0
