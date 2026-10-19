"""Train CMPANet-S with shared norm1. See --help for data and teacher paths."""
import sys
from pathlib import Path
from main import main

if __name__ == '__main__':
    args = sys.argv[1:]
    if not any(x == '--configs' or x.startswith('--configs=') for x in args):
        args = ['--configs', str(Path(__file__).parent / 'configs/cmpanet_small_head24.json')] + args
    main(args)
