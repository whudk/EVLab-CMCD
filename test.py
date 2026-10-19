"""Evaluate a CMPANet decoder plus its frozen image encoder on a labeled manifest."""
import argparse
import copy
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent


def metrics(cm):
    tn, fp, fn, tp = [int(x) for row in cm for x in row]
    def percent(a, b):
        return 100.0 * a / b if b else 0.0
    return dict(precision=percent(tp, tp+fp), recall=percent(tp, tp+fn),
                f1=percent(2*tp, 2*tp+fp+fn), iou_change=percent(tp, tp+fp+fn),
                oa=percent(tn+tp, tn+fp+fn+tp),
                miou=(percent(tn, tn+fp+fn)+percent(tp, tp+fp+fn))/2)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--configs', default=str(ROOT/'configs/cmpanet_small_head24.json'))
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--pretrained', required=True, help='Matching frozen image teacher checkpoint.')
    p.add_argument('--valtxt', required=True, help='map image label paths; relative paths use the current working directory.')
    p.add_argument('--output', default='outputs/cmpanet_small_head24')
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--memory-fraction', type=float, default=0.60)
    p.add_argument('--save-predictions', action='store_true')
    p.add_argument('--expect-samples', type=int)
    args = p.parse_args()
    if not 0 < args.memory_fraction <= 1 or args.workers < 0 or args.gpu < 0:
        p.error('Use a positive memory fraction <= 1, nonnegative workers and GPU index.')
    output = Path(args.output)
    if (output/'results.json').exists() or (output/'samples.jsonl').exists():
        p.error('Output already contains evaluation records; use a new --output directory.')
    cfg = json.loads(Path(args.configs).read_text(encoding='utf-8'))
    manifest = Path(args.valtxt).resolve()
    lines = manifest.read_text(encoding='utf-8-sig').splitlines()
    if not lines:
        p.error('Empty validation manifest.')
    for i, line in enumerate(lines, 1):
        paths = line.split()
        if len(paths) != 3 or not all(Path(x).is_file() for x in paths):
            p.error(f'Invalid map/image/label paths on manifest line {i}.')
    if args.expect_samples is not None and len(lines) != args.expect_samples:
        p.error(f'Expected {args.expect_samples} samples, found {len(lines)}.')
    if cfg['network']['decoder']['head']['params'].get('feature_normalization') != 'shared_chunk_norm1':
        p.error('This evaluator requires the verified shared_chunk_norm1 normalization.')
    for path in (args.checkpoint, args.pretrained):
        if not Path(path).is_file():
            p.error(f'Missing checkpoint: {path}')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    import torch
    from torch.nn import functional as F
    from lib.checkpoint import load_checkpoint, load_decoder, sha256
    from lib.models.model_manager import ModelManager
    from lib.datasets.data_loader import DataLoader
    from lib.utils.tools.configer import Configer
    if not torch.cuda.is_available():
        p.error('A CUDA-enabled PyTorch installation and GPU are required.')
    torch.set_num_threads(2)
    torch.manual_seed(304)
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg['data'].update(valtxt=str(manifest), workers=args.workers)
    cfg['val']['batch_size'] = 1
    cfg['network']['image_encoder']['pretrained'] = str(Path(args.pretrained).resolve())
    output.mkdir(parents=True, exist_ok=True)
    manager = ModelManager(Configer(config_dict=copy.deepcopy(cfg)))
    decoder = manager.build_decoder()
    loading = load_decoder(decoder, args.checkpoint, cfg)
    enc = cfg['network']['image_encoder']
    teacher = manager.build_encoder(encoder_name=enc['name'], net_params=copy.deepcopy(enc['params']))
    teacher_state = load_checkpoint(args.pretrained)['teacher']
    teacher_state = {k.removeprefix('module.'): v for k, v in teacher_state.items()}
    teacher.load_state_dict({k.removeprefix('backbone.'): v for k, v in teacher_state.items()
                            if k.startswith('backbone.')}, strict=True)
    decoder.cuda().eval()
    teacher.cuda().eval()
    loader = DataLoader(Configer(config_dict=cfg)).get_valloader()
    start = time.time()
    cm = torch.zeros(2, 2, dtype=torch.int64)
    ignored = 0
    with torch.inference_mode(), (output/'samples.jsonl').open('w', encoding='utf-8') as stream:
        for i, (inputs, target, masks, meta) in enumerate(loader):
            left, right = inputs.cuda().split(3, dim=1)
            feats = [f[:, 1:, :] for f in teacher.get_feats(right)]
            # The label-derived mask affects only the auxiliary alignment branch.
            # The segmentation branch encodes the map independently (see cd_head.py).
            logits = decoder(left, feats, mask=masks.cuda(), with_embed=False)['pred']
            pred = F.interpolate(logits, target.shape[-2:], mode='bilinear', align_corners=True).argmax(1).cpu()
            valid = (target >= 0) & (target < 2)
            sample_cm = torch.bincount(2*target[valid].long()+pred[valid], minlength=4).reshape(2, 2)
            cm += sample_cm
            sample_ignored = int((~valid).sum())
            ignored += sample_ignored
            stream.write(json.dumps(dict(index=i, name=Path(meta['old_path'][0]).name,
                confusion_matrix=sample_cm.tolist(), valid_pixels=int(valid.sum()), ignored_pixels=sample_ignored))+'\n')
            stream.flush()
            if args.save_predictions:
                from PIL import Image
                directory = output/'predictions'
                directory.mkdir(exist_ok=True)
                pixels = (pred[0].numpy()*255).astype('uint8')
                Image.fromarray(pixels).save(directory/f'{i:05d}.png')
            if i == 0 or (i+1) % 50 == 0 or i+1 == len(loader):
                print(f'Evaluated {i+1}/{len(loader)} samples', flush=True)
    result = dict(samples=len(loader.dataset), confusion_matrix=cm.tolist(), metrics_percent=metrics(cm.tolist()),
        valid_pixels=int(cm.sum()), ignored_pixels=ignored, total_pixels=int(cm.sum())+ignored,
        valid_change_pixels=int(cm[1].sum()), checkpoint_sha256=sha256(args.checkpoint),
        teacher_sha256=sha256(args.pretrained), manifest_sha256=sha256(manifest),
        checkpoint=Path(args.checkpoint).name, teacher=Path(args.pretrained).name,
        normalization='shared_chunk_norm1', precision='FP32; TF32 disabled', batch_size=1,
        decoder_loading=loading, teacher_loading='strict backbone', elapsed_seconds=time.time()-start)
    (output/'results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    (output/'evaluation_config.json').write_text(json.dumps(cfg, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
