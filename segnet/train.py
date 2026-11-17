"""Trains MorphSymNet on the half-resolution Cityscapes cache and reports the validation mIoU.

    python -m segnet.build_cache                        # once
    python -m segnet.train --features rgb,morph,sym --name full
    python -m segnet.train --features rgb --name rgb    # baseline: same network, RGB input

All runs with the same --seed see the same crops in the same order, so runs that differ only
in --features are paired comparisons.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from .build_cache import CLASSES, IGNORE
from .model import build_model

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sample_batch(images, labels, indices, crop, scale_range, rng, device):
    """Random scale + crop + horizontal flip. Returns RGB (B, 3, h, w) in [0, 1] and labels."""
    H, W = images.shape[1:3]
    ch, cw = crop
    s = rng.uniform(*scale_range)                       # one scale per batch
    rh, rw = min(H, int(round(ch / s))), min(W, int(round(cw / s)))
    xs, ys = [], []
    for i in indices:
        y0, x0 = rng.integers(0, H - rh + 1), rng.integers(0, W - rw + 1)
        xs.append(images[i, y0:y0 + rh, x0:x0 + rw])
        ys.append(labels[i, y0:y0 + rh, x0:x0 + rw])
    x = torch.from_numpy(np.stack(xs)).to(device).permute(0, 3, 1, 2).float() / 255.0
    y = torch.from_numpy(np.stack(ys)).to(device)
    if (rh, rw) != (ch, cw):
        x = F.interpolate(x, size=crop, mode='bilinear', align_corners=False).clamp_(0, 1)
        y = F.interpolate(y[:, None].float(), size=crop, mode='nearest')[:, 0].to(torch.uint8)
    flip = torch.from_numpy(rng.random(len(indices)) < 0.5).to(device)
    x = torch.where(flip[:, None, None, None], x.flip(3), x)
    y = torch.where(flip[:, None, None], y.flip(2), y)
    return x.contiguous(), y.long()


def photometric(x, rng, gen):
    """Random brightness, contrast, gamma, saturation and noise, drawn per image.
    x: (B, 3, h, w) in [0, 1]. rng draws the settings, gen (a torch generator) the noise; both
    are separate from the generator of the crops, so the crops stay the same with and
    without this step."""
    B = x.shape[0]

    def draw(lo, hi):
        return torch.from_numpy(rng.uniform(lo, hi, B).astype(np.float32)).to(x.device).view(B, 1, 1, 1)

    grey = x.mean(dim=1, keepdim=True)
    x = grey + (x - grey) * draw(0.7, 1.3)                         # saturation
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    x = (x - mean) * draw(0.75, 1.3) + mean + draw(-0.12, 0.12)    # contrast, brightness
    x = x.clamp(0, 1) ** torch.exp(draw(-0.35, 0.35))              # gamma 0.7 .. 1.42
    sigma = draw(0.0, 0.03) * (draw(0, 1) < 0.5)                   # noise on half of the images
    x = x + sigma * torch.randn(x.shape, generator=gen, device=x.device)
    return ((x.clamp(0, 1) * 255).round() / 255).contiguous()      # 8 bits, like a stored image


@torch.no_grad()
def evaluate(model, images, labels, device, batch=10, transform=None):
    """Confusion matrix over the whole validation set -> per-class IoU, mIoU, pixel accuracy.
    transform(x, i): optional change of the RGB batch that starts at image i (robustness.py)."""
    model.eval()
    n = len(CLASSES)
    conf = torch.zeros(n, n, dtype=torch.long, device=device)
    for i in range(0, len(images), batch):
        x = torch.from_numpy(images[i:i + batch]).to(device).permute(0, 3, 1, 2).float() / 255.0
        y = torch.from_numpy(labels[i:i + batch]).to(device).long()
        with torch.cuda.amp.autocast():
            if transform is not None:
                x = transform(x, i)
            pred = model(x.contiguous()).argmax(1)
        keep = y != IGNORE
        conf += torch.bincount(y[keep] * n + pred[keep], minlength=n * n).view(n, n)
    conf = conf.double()
    iou = conf.diag() / (conf.sum(0) + conf.sum(1) - conf.diag()).clamp_min(1)
    return {'miou': iou.mean().item(), 'acc': (conf.diag().sum() / conf.sum()).item(),
            'iou': dict(zip(CLASSES, iou.tolist()))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--features', default='rgb,morph,sym')
    p.add_argument('--name', default=None)
    p.add_argument('--cache', default=os.path.join(ROOT, 'cache'))
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--crop', type=int, nargs=2, default=(384, 768))
    p.add_argument('--scale', type=float, nargs=2, default=(0.75, 1.5))
    p.add_argument('--lr', type=float, default=2e-3)
    p.add_argument('--wd', type=float, default=1e-2)
    p.add_argument('--warmup', type=int, default=300)
    p.add_argument('--eval-every', type=int, default=5)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--input-norm', action='store_true', help='BatchNorm on the feature stack')
    p.add_argument('--gate', action='store_true',
                   help='learnable scalar per non-RGB plane, initialised to 0')
    p.add_argument('--inject', default='input', choices=('input', 'quarter'),
                   help='where the non-RGB planes enter: first convolution, or a side stem at 1/4')
    p.add_argument('--morph-planes', default='all', choices=('all', 'range'))
    p.add_argument('--sym-half-widths', type=int, nargs='+', default=None)
    p.add_argument('--diff-bank', default='full', help='difference-filter bank, see segnet/diff_bank.py')
    p.add_argument('--augment', action='store_true',
                   help='photometric augmentation: brightness, contrast, gamma, saturation, noise')
    p.add_argument('--guided-up', action='store_true',
                   help='image-guided upsampling of the logits (segnet/guided_upsample.py)')
    p.add_argument('--ema', type=float, default=0.0,
                   help='also evaluate an exponential moving average of the weights with this decay (e.g. 0.999)')
    p.add_argument('--learn-diff', action='store_true',
                   help='add LearnableDiffFilter: generalFilter with trained parameters')
    p.add_argument('--ld-stride', type=int, default=2, choices=(2, 4), help='resolution of its planes')
    p.add_argument('--ld-spacings', type=float, nargs='+', default=(2.0,),
                   help='initial circle distances; the 8 presets are created once per value')
    p.add_argument('--ld-directions', type=int, default=8)
    p.add_argument('--ld-freeze', action='store_true', help='keep the filters at their initial setting')
    p.add_argument('--ld-lr', type=float, default=1.0, help='learning-rate factor of the filter parameters')
    p.add_argument('--subset', type=int, default=0,
                   help='train on a fixed random subset of N images (low-data regime)')
    p.add_argument('--limit', type=int, default=0, help='use only the first N train images (smoke test)')
    a = p.parse_args()

    feats = tuple(a.features.split(','))
    tags = (('_norm' if a.input_norm else '') + ('_gate' if a.gate else '') + ('_quarter' if a.inject == 'quarter' else '')
            + ('_range' if a.morph_planes == 'range' else '')
            + ('_hw' + '-'.join(map(str, a.sym_half_widths)) if a.sym_half_widths else '')
            + (f'_bank-{a.diff_bank}' if 'diff' in feats and a.diff_bank != 'full' else '')
            + (('_ldfrozen' if a.ld_freeze else '_ldiff') + (f'_stride{a.ld_stride}' if a.ld_stride != 2 else '')
               + ('_sp' + '-'.join(f'{v:g}' for v in a.ld_spacings) if tuple(a.ld_spacings) != (2.0,) else '')
               + (f'_lr{a.ld_lr:g}' if a.ld_lr != 1.0 else '') if a.learn_diff else '')
            + ('_gup' if a.guided_up else '') + ('_aug' if a.augment else '')
            + (f'_ep{a.epochs}' if a.epochs != 20 else '') + (f'_n{a.subset}' if a.subset else ''))
    name = a.name or '+'.join(feats) + tags + f'_s{a.seed}'
    out_dir = os.path.join(ROOT, 'runs', name)
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device('cuda')
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    torch.backends.cudnn.benchmark = True
    aug_rng = np.random.default_rng(a.seed + 1000)
    aug_gen = torch.Generator(device=device).manual_seed(a.seed + 1000)

    tr_img, tr_lbl = np.load(f'{a.cache}/train_img.npy'), np.load(f'{a.cache}/train_lbl.npy')
    va_img, va_lbl = np.load(f'{a.cache}/val_img.npy'), np.load(f'{a.cache}/val_lbl.npy')
    if a.limit:
        tr_img, tr_lbl = tr_img[:a.limit], tr_lbl[:a.limit]
    if a.subset:
        # the same images for every run, whatever its seed
        keep = np.sort(np.random.default_rng(12345).choice(len(tr_img), a.subset, replace=False))
        tr_img, tr_lbl = tr_img[keep], tr_lbl[keep]

    model = build_model(len(CLASSES), feats, vars(a)).to(device)
    n_params = sum(q.numel() for q in model.parameters())
    print(f'{name}: {model.frontend.n_features} input planes {model.frontend.groups}, {n_params / 1e6:.2f} M parameters')

    # the filter parameters are not weight-decayed: 0 is not their neutral value
    filt = [q for q in model.ldiff.parameters() if q.requires_grad] if model.ldiff is not None else []
    if model.gup is not None:
        filt += list(model.gup.parameters())
    rest = [q for q in model.parameters() if q.requires_grad and all(q is not f for f in filt)]
    decay = [q for q in rest if q.ndim > 1]
    no_decay = [q for q in rest if q.ndim <= 1]
    opt = torch.optim.AdamW([{'params': decay, 'weight_decay': a.wd},
                             {'params': no_decay, 'weight_decay': 0.0}]
                            + ([{'params': filt, 'weight_decay': 0.0, 'lr': a.lr * a.ld_lr}] if filt else []),
                            lr=a.lr)
    steps_per_epoch = len(tr_img) // a.batch
    total = a.epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda it: min(1.0, (it + 1) / a.warmup) * max(0.0, 1 - it / total) ** 0.9)
    scaler = torch.cuda.amp.GradScaler()

    if model.ldiff is not None:
        log_names = model.ldiff.names()
    log = {'name': name, 'features': feats, 'args': vars(a), 'n_params': n_params,
           'n_planes': model.frontend.n_features, 'epochs': []}
    if model.ldiff is not None:
        log['ldiff_names'] = log_names
        log['ldiff_init'] = model.ldiff.state()
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()} if a.ema else None
    t_start = time.time()
    for epoch in range(1, a.epochs + 1):
        model.train()
        order = rng.permutation(len(tr_img))
        loss_sum = 0.0
        for it in range(steps_per_epoch):
            x, y = sample_batch(tr_img, tr_lbl, order[it * a.batch:(it + 1) * a.batch],
                                tuple(a.crop), a.scale, rng, device)
            if a.augment:
                x = photometric(x, aug_rng, aug_gen)
            with torch.cuda.amp.autocast():
                loss = F.cross_entropy(model(x).float(), y, ignore_index=IGNORE)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            if ema is not None:
                with torch.no_grad():
                    for k, v in model.state_dict().items():
                        if v.dtype.is_floating_point:
                            ema[k].lerp_(v, 1 - a.ema)
                        else:
                            ema[k].copy_(v)
            loss_sum += loss.item()
        entry = {'epoch': epoch, 'loss': loss_sum / steps_per_epoch, 'time': time.time() - t_start}
        if epoch % a.eval_every == 0 or epoch == a.epochs:
            entry.update(evaluate(model, va_img, va_lbl, device))
            if ema is not None:
                raw = {k: v.clone() for k, v in model.state_dict().items()}
                model.load_state_dict(ema)
                entry['ema'] = evaluate(model, va_img, va_lbl, device)
                model.load_state_dict(raw)
                print(f"           averaged weights: mIoU {100 * entry['ema']['miou']:.2f}", flush=True)
            print(f"epoch {epoch:3d}  loss {entry['loss']:.4f}  mIoU {100 * entry['miou']:.2f}  "
                  f"acc {100 * entry['acc']:.2f}  {entry['time'] / 60:.1f} min", flush=True)
        else:
            print(f"epoch {epoch:3d}  loss {entry['loss']:.4f}  {entry['time'] / 60:.1f} min", flush=True)
        if model.gate is not None:
            entry['gate'] = model.gate.detach().float().tolist()
        if model.gup is not None:
            entry['gup'] = model.gup.state()
        if model.ldiff is not None and not a.ld_freeze:
            entry['ldiff'] = model.ldiff.state()
        log['epochs'].append(entry)
        with open(os.path.join(out_dir, 'log.json'), 'w') as f:
            json.dump(log, f, indent=1)
    torch.save(model.state_dict(), os.path.join(out_dir, 'model.pt'))
    if ema is not None:
        torch.save(ema, os.path.join(out_dir, 'model_ema.pt'))


if __name__ == '__main__':
    main()
