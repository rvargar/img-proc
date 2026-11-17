"""Validation mIoU of trained runs under changes of the input image.

    python -m segnet.robustness                       # all full-set runs in runs/
    python -m segnet.robustness --runs rgb_s0 rgb+diff_s0

The image is changed in [0, 1] before the feature front-end, clipped and rounded to 8 bits,
so every model sees exactly the same changed images. Writes runs/robustness.json.
"""
import argparse
import json
import os

import numpy as np
import torch

from .build_cache import CLASSES
from .model import build_model
from .train import ROOT, evaluate


def _noise(sigma):
    def f(x, i):
        g = torch.Generator(device=x.device).manual_seed(1000 + i)
        return x + sigma * torch.randn(x.shape, generator=g, device=x.device)
    return f


def _contrast(k):
    def f(x, i):
        m = x.mean(dim=(1, 2, 3), keepdim=True)
        return (x - m) * k + m
    return f


CONDITIONS = {
    'clean': lambda x, i: x,
    'brightness -0.2': lambda x, i: x - 0.2,
    'brightness -0.1': lambda x, i: x - 0.1,
    'brightness +0.1': lambda x, i: x + 0.1,
    'brightness +0.2': lambda x, i: x + 0.2,
    'contrast x0.5': _contrast(0.5),
    'contrast x0.75': _contrast(0.75),
    'contrast x1.5': _contrast(1.5),
    'gamma 0.5': lambda x, i: x ** 0.5,
    'gamma 0.7': lambda x, i: x ** 0.7,
    'gamma 1.5': lambda x, i: x ** 1.5,
    'gamma 2.0': lambda x, i: x ** 2.0,
    'noise 0.02': _noise(0.02),
    'noise 0.05': _noise(0.05),
}


def quantised(f):
    return lambda x, i: (f(x, i).clamp(0, 1) * 255).round() / 255


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs', nargs='*', default=None)
    p.add_argument('--cache', default=os.path.join(ROOT, 'cache'))
    a = p.parse_args()
    runs_dir = os.path.join(ROOT, 'runs')
    device = torch.device('cuda')
    va_img, va_lbl = np.load(f'{a.cache}/val_img.npy'), np.load(f'{a.cache}/val_lbl.npy')

    out_path = os.path.join(runs_dir, 'robustness.json')
    results = json.load(open(out_path)) if os.path.exists(out_path) else {}
    names = a.runs or sorted(d for d in os.listdir(runs_dir)
                             if os.path.exists(os.path.join(runs_dir, d, 'model.pt')))
    for name in names:
        log = json.load(open(os.path.join(runs_dir, name, 'log.json')))
        if log['args'].get('subset') or log['args'].get('limit'):
            continue
        model = build_model(len(CLASSES), log['features'], log['args']).to(device)
        model.load_state_dict(torch.load(os.path.join(runs_dir, name, 'model.pt'), map_location=device))
        results[name] = {c: evaluate(model, va_img, va_lbl, device, transform=quantised(f))['miou']
                         for c, f in CONDITIONS.items()}
        print(name, ' '.join(f'{100 * v:.1f}' for v in results[name].values()), flush=True)
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=1)

    # table: configurations (seed-averaged) x conditions, as mIoU and as change to clean
    groups = {}
    for name, r in results.items():
        groups.setdefault(name.rsplit('_s', 1)[0], []).append(r)
    conds = list(CONDITIONS)
    mean = {g: {c: 100 * np.mean([r[c] for r in rs]) for c in conds} for g, rs in groups.items()}
    order = sorted(mean, key=lambda g: -mean[g]['clean'])
    for title, val in (('mIoU %', lambda g, c: mean[g][c]),
                       ('change to clean, points', lambda g, c: mean[g][c] - mean[g]['clean'])):
        print(f'\n{title} (seed-averaged)')
        print(f"{'configuration':26s}" + ''.join(f'{c.replace("brightness", "br").replace("contrast", "con"):>11s}' for c in conds)
              + f"{'mean chg':>10s}")
        for g in order:
            changed = np.mean([mean[g][c] for c in conds[1:]]) - (mean[g]['clean'] if 'change' in title else 0)
            print(f'{g:26s}' + ''.join(f'{val(g, c):11.1f}' for c in conds) + f'{changed:10.1f}')


if __name__ == '__main__':
    main()
