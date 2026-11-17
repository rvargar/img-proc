"""Builds a half-resolution (512 x 1024) uint8 cache of Cityscapes for fast experiments.

    python -m segnet.build_cache

Reads  <root>/cityscapes/{train,val}/*.png  (images) and  <root>/gtFine/{train,val}/*.png
(labelIds masks with the same file names) and writes, per split, to cache/:
    <split>_img.npy  (n, 512, 1024, 3) uint8 RGB
    <split>_lbl.npy  (n, 512, 1024)    uint8 trainIds, 255 = ignore
    <split>_names.txt

Images whose mask has no labelled pixel are skipped: the train folder also holds the
Cityscapes TEST cities (berlin, bielefeld, bonn, leverkusen, mainz, munich), whose public
masks are empty.
"""
import argparse
import os
from multiprocessing import Pool

import cv2
import numpy as np

# Cityscapes labelId -> trainId (19 evaluation classes), everything else is ignored
CLASSES = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
           'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck',
           'bus', 'train', 'motorcycle', 'bicycle']
LABEL_IDS = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
IGNORE = 255
LUT = np.full(256, IGNORE, dtype=np.uint8)
LUT[LABEL_IDS] = np.arange(len(LABEL_IDS), dtype=np.uint8)

SIZE = (1024, 512)   # (width, height)


def load(args):
    img_path, lbl_path = args
    lbl = LUT[cv2.imread(lbl_path, cv2.IMREAD_UNCHANGED)]
    if not (lbl != IGNORE).any():
        return None
    img = cv2.cvtColor(cv2.imread(img_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return (cv2.resize(img, SIZE, interpolation=cv2.INTER_AREA),
            cv2.resize(lbl, SIZE, interpolation=cv2.INTER_NEAREST))


def build(root, out_dir, split, workers):
    img_dir, lbl_dir = os.path.join(root, 'cityscapes', split), os.path.join(root, 'gtFine', split)
    names = sorted(n for n in os.listdir(img_dir) if os.path.exists(os.path.join(lbl_dir, n)))
    jobs = [(os.path.join(img_dir, n), os.path.join(lbl_dir, n)) for n in names]
    with Pool(workers) as pool:
        results = pool.map(load, jobs, chunksize=8)
    kept = [(n, r) for n, r in zip(names, results) if r is not None]
    np.save(os.path.join(out_dir, f'{split}_img.npy'), np.stack([r[0] for _, r in kept]))
    np.save(os.path.join(out_dir, f'{split}_lbl.npy'), np.stack([r[1] for _, r in kept]))
    with open(os.path.join(out_dir, f'{split}_names.txt'), 'w') as f:
        f.write('\n'.join(n for n, _ in kept) + '\n')
    print(f'{split}: kept {len(kept)} of {len(names)} images')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='/home/rvarga/Cityscapes')
    p.add_argument('--out', default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cache'))
    p.add_argument('--workers', type=int, default=12)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for s in ('val', 'train'):
        build(a.root, a.out, s, a.workers)
