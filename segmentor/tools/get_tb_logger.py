from datetime import datetime
from pathlib import Path
from tensorboardX import SummaryWriter


def get_tb_logger(configer):
    root = configer.get("logging", "tb_logger")
    if not root:
        return None
    name = configer.get("logging", "tb_name")
    return SummaryWriter(
        str(Path(root) / (name + "_" + datetime.now().strftime("%Y%m%d_%H%M%S")))
    )
