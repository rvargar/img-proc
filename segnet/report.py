"""Summarises the runs in runs/: validation mIoU per configuration, learning curves and
per-class IoU. Runs on the full training set and runs on a subset are reported separately.

    python -m segnet.report
"""
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def final(log, key):
    return 100 * log['epochs'][-1][key]


def table(runs):
    width = max(len(k) for k in runs) + 2
    base = runs.get('rgb')
    base_miou = np.mean([final(l, 'miou') for l in base]) if base else None

    print(f"{'configuration':{width}s} {'planes':>6s} {'seeds':>5s} {'mIoU %':>7s} {'vs rgb':>7s}  "
          f"{'per seed':18s} {'acc %':>6s} {'train min':>9s}")
    for key, logs in sorted(runs.items(), key=lambda kv: -np.mean([final(l, 'miou') for l in kv[1]])):
        miou = [final(l, 'miou') for l in logs]
        delta = f"{np.mean(miou) - base_miou:+.2f}" if base and key != 'rgb' else ''
        print(f"{key:{width}s} {logs[0]['n_planes']:6d} {len(logs):5d} {np.mean(miou):7.2f} {delta:>7s}  "
              f"{', '.join(f'{m:.2f}' for m in miou):18s} {np.mean([final(l, 'acc') for l in logs]):6.2f} "
              f"{np.mean([l['epochs'][-1]['time'] / 60 for l in logs]):9.1f}")

    print("\nmIoU % during training (seed-averaged):")
    for key, logs in runs.items():
        curve = defaultdict(list)
        for l in logs:
            for e in l['epochs']:
                if 'miou' in e:
                    curve[e['epoch']].append(100 * e['miou'])
        print(f"  {key:{width}s} " + "  ".join(f"ep{ep}: {np.mean(v):.1f}" for ep, v in sorted(curve.items())))

    if base:
        print("\nper-class IoU %: rgb, then the difference of each configuration to rgb (seed-averaged)")
        others = [k for k in runs if k != 'rgb']
        print(f"  {'class':14s} {'rgb':>6s}  " + "  ".join(f"{k[-18:]:>18s}" for k in others))
        for c in base[0]['epochs'][-1]['iou']:
            b = np.mean([100 * l['epochs'][-1]['iou'][c] for l in base])
            cells = [f"{np.mean([100 * l['epochs'][-1]['iou'][c] for l in runs[k]]) - b:+18.1f}" for k in others]
            print(f"  {c:14s} {b:6.1f}  " + "  ".join(cells))


def learned_filters(log):
    """Initial -> final parameters of the LearnableDiffFilter of one run."""
    init, last, names = log['ldiff_init'], log['epochs'][-1]['ldiff'], log['ldiff_names']

    def row(values, fmt='{:+.2f}'):
        return ' '.join(fmt.format(v) for v in np.ravel(values))

    print(f"\nlearned difference filters of {log['name']}: initial -> final")
    print("  colour r g b | radii px | axis ratio, rotation deg | gamma: v, c, |v-c| | beta: self, "
          "partner (angular radial opposite), |a-b| (same) | temperatures stat, alpha, alpha2 | rho")
    for i, name in enumerate(names):
        for tag, s in (('init ', init), ('final', last)):
            print(f"  {name if tag == 'init ' else '':16s} {tag} {row(s['colour'][i])} | {row(s['radius'][i], '{:5.2f}')} | "
                  f"{np.exp(2 * s['log_ratio'][i]):.2f} {np.degrees(s['angle'][i]):+5.1f} | {row(s['gamma'][i])} | "
                  f"{s['beta_self'][i]:+.2f}  {row(s['beta_partner'][i])}  {row(s['beta_abs'][i])} | "
                  f"{row([20 * s[k][i] for k in ('t_stat', 't_alpha', 't_alpha2')], '{:+6.1f}')} | {s['rho'][i]:+.2f}")


def main():
    groups = defaultdict(lambda: defaultdict(list))        # subset size -> configuration -> logs
    for path in sorted(glob.glob(os.path.join(ROOT, 'runs', '*', 'log.json'))):
        log = json.load(open(path))
        last = log['epochs'][-1]
        if 'miou' not in last or last['epoch'] != log['args']['epochs']:
            continue                                       # unfinished run
        subset = log['args'].get('subset', 0)
        config = re.sub(r'_s\d+$', '', log['name']).replace(f'_n{subset}', '')
        groups[subset][config].append(log)
    if not groups:
        print("no finished runs")
    for subset in sorted(groups):
        runs = groups[subset]
        any_log = next(iter(runs.values()))[0]
        title = f"{subset} training images" if subset else "full training set"
        print(f"\n===== {title}, {any_log['args']['epochs']} epochs =====")
        table(runs)
        for logs in runs.values():
            for log in logs:
                if 'ldiff' in log['epochs'][-1]:
                    learned_filters(log)


if __name__ == '__main__':
    main()
