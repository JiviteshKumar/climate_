# -*- coding: utf-8 -*-
"""
dl_common.py -- shared dataset loading for the Deep Learning laboratory.

Dataset: PerceptiLabs Weather-Analysis (Multi-class Weather Dataset)
         https://github.com/PerceptiLabs/Weather-Analysis
         1124 RGB images, 224x224, four classes: cloud, rain, shine, sunrise.

The dataset is the image corpus behind the Climate Intelligence use case
(KJS-CES-01): classifying the sky condition in a photograph is the vision
counterpart of the region-wise heatwave severity problem.

Jivitesh Kumar | Roll No. 16010423041 | Batch A2 | KJSSE, Sem VII
"""

import os
import glob
import numpy as np
from PIL import Image

ROOT = os.environ.get(
    "WEATHER_DIR",
    r"C:\Users\jivit\AppData\Local\Temp\claude\C--Users-jivit-Downloads-porj"
    r"\638c18fa-5dac-4553-b5c4-95a35618b111\scratchpad\dldata\Weather-Analysis")

CLASSES = ["cloud", "rain", "shine", "sunrise"]
SEED = 42


def list_files():
    """Return (paths, labels) for every image in the dataset, sorted for determinism."""
    paths, labels = [], []
    for idx, c in enumerate(CLASSES):
        fs = sorted(glob.glob(os.path.join(ROOT, c, "*")))
        fs = [f for f in fs if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        paths += fs
        labels += [idx] * len(fs)
    return paths, np.array(labels, dtype=np.int64)


def load_images(size=48, gray=False, verbose=True):
    """Load every image, resized to size x size, scaled to [0, 1]."""
    paths, y = list_files()
    n = len(paths)
    ch = 1 if gray else 3
    X = np.zeros((n, size, size, ch), dtype=np.float32)
    for i, p in enumerate(paths):
        im = Image.open(p).convert("L" if gray else "RGB").resize((size, size), Image.BILINEAR)
        X[i] = np.asarray(im, dtype=np.float32).reshape(size, size, ch) / 255.0
        if verbose and (i + 1) % 300 == 0:
            print("    loaded %4d / %d" % (i + 1, n))
    return X, y


def handcrafted_features(verbose=True):
    """Eight interpretable features per image, for the from-scratch network.

    The features are chosen for physical meaning on weather photographs rather
    than for raw accuracy: a sunrise is warm and strongly graded top to bottom,
    an overcast sky is bright, flat and desaturated, and rain is dark and
    low-contrast.
    """
    paths, y = list_files()
    names = ["mean_R", "mean_G", "mean_B", "brightness",
             "saturation", "contrast", "sky_ground_grad", "warm_ratio"]
    F = np.zeros((len(paths), len(names)), dtype=np.float64)
    for i, p in enumerate(paths):
        a = np.asarray(Image.open(p).convert("RGB").resize((64, 64), Image.BILINEAR),
                       dtype=np.float64) / 255.0
        r, g, b = a[..., 0], a[..., 1], a[..., 2]
        lum = 0.299 * r + 0.587 * g + 0.114 * b
        mx, mn = a.max(axis=2), a.min(axis=2)
        sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
        top, bot = lum[:21].mean(), lum[43:].mean()
        F[i] = [r.mean(), g.mean(), b.mean(), lum.mean(), sat.mean(), lum.std(),
                top - bot, r.mean() / max(b.mean(), 1e-6)]
        if verbose and (i + 1) % 300 == 0:
            print("    featurised %4d / %d" % (i + 1, len(paths)))
    return F, y, names


def stratified_split(y, train=0.70, val=0.15, seed=SEED):
    """Class-balanced train / validation / test index split."""
    rng = np.random.default_rng(seed)
    tr, va, te = [], [], []
    for c in range(len(CLASSES)):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n = len(idx)
        a, b = int(n * train), int(n * (train + val))
        tr += list(idx[:a]); va += list(idx[a:b]); te += list(idx[b:])
    rng.shuffle(tr); rng.shuffle(va); rng.shuffle(te)
    return np.array(tr), np.array(va), np.array(te)


def banner(t, ch="="):
    print("\n" + ch * 76)
    print(" " + t)
    print(ch * 76)


def class_table():
    _, y = list_files()
    print(" %-10s %8s %9s" % ("CLASS", "IMAGES", "SHARE"))
    print(" " + "-" * 30)
    for i, c in enumerate(CLASSES):
        n = int((y == i).sum())
        print(" %-10s %8d %8.1f%%" % (c, n, 100.0 * n / len(y)))
    print(" %-10s %8d %8.1f%%" % ("TOTAL", len(y), 100.0))
