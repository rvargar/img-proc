"""Runs a trained MorphSymNet on validation images and writes a montage:
image | ground truth | prediction, one row per image.

    python -m segnet.predict --run rgb+morph+sym_s0 --indices 0 100 250 --out docs/segnet_predictions.jpg
"""
import argparse
import json
import os

import cv2
import numpy as np
import torch

from .build_cache import IGNORE
from .model import build_model

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Cityscapes colours of the 19 classes, RGB
PALETTE = np.array([[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], [190, 153, 153],
                    [153, 153, 153], [250, 170, 30], [220, 220, 0], [107, 142, 35], [152, 251, 152],
                    [70, 130, 180], [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70],
                    [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]], dtype=np.uint8)


def colourise(labels):
    out = np.zeros(labels.shape + (3,), dtype=np.uint8)
    valid = labels != IGNORE
    out[valid] = PALETTE[labels[valid]]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    p.add_argument('--indices', type=int, nargs='+', default=[0, 100, 250])
    p.add_argument('--cache', default=os.path.join(ROOT, 'cache'))
    p.add_argument('--out', default=None)
    a = p.parse_args()

    run_dir = os.path.join(ROOT, 'runs', a.run)
    log = json.load(open(os.path.join(run_dir, 'log.json')))
    model = build_model(len(PALETTE), log['features'], log['args']).cuda().eval()
    model.load_state_dict(torch.load(os.path.join(run_dir, 'model.pt')))

    images = np.load(f'{a.cache}/val_img.npy', mmap_mode='r')
    labels = np.load(f'{a.cache}/val_lbl.npy', mmap_mode='r')
    rows = []
    for i in a.indices:
        img = np.ascontiguousarray(images[i])
        x = torch.from_numpy(img).cuda().permute(2, 0, 1)[None].float() / 255.0
        with torch.no_grad(), torch.cuda.amp.autocast():
            pred = model(x.contiguous()).argmax(1)[0].cpu().numpy().astype(np.uint8)
        rows.append(np.hstack([img, colourise(np.asarray(labels[i])), colourise(pred)]))
    montage = cv2.resize(np.vstack(rows), None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    out = a.out or os.path.join(run_dir, 'predictions.jpg')
    cv2.imwrite(out, cv2.cvtColor(montage, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print("wrote", out)


if __name__ == '__main__':
    main()
