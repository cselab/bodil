#!/usr/bin/env python3
import sys
import scipy
import numpy as np
import itertools
import re
import matplotlib.pylab as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-t1', type=float, default=None)
parser.add_argument('-t0', type=float, default=None)
parser.add_argument('-c', action='store_true', help='use common colorbar')
parser.add_argument('files', nargs='+', help='List of .mat files to process')
args = parser.parse_args()

plt.rcParams['image.cmap'] = 'jet'
skip = set(["__header__", "__version__", "__globals__"])
field = "A", "u", "P", "t", "x"
scales = 1000, 100, 1
labels = "A (mm/s)", "u (cm/s)", "P (Pa)"
vmin = {}
vmax = {}
for path in args.files:
    data = scipy.io.loadmat(path)
    for k, v in data.items():
        if k not in skip:
            if np.size(v):
                if v.ndim == 3:
                    for f, s, d in zip(field, scales, np.rollaxis(v, 2)):
                        fmin = np.min(s * d)
                        fmax = np.max(s * d)
                        vmin[f] = min(fmin, vmin[f]) if f in vmin else fmin
                        vmax[f] = max(fmax, vmax[f]) if f in vmax else fmax
for path in args.files:
    data = scipy.io.loadmat(path)
    for k, v in data.items():
        if k not in skip:
            if np.size(v):
                if v.ndim == 3:
                    v[:, :, field.index("t")] *= 1000
                    v[:, :, field.index("x")] *= 1000
                    t0 = v[0, 0, field.index("t")]
                    t1 = v[0, -1, field.index("t")]
                    x0 = v[0, 0, field.index("x")]
                    x1 = v[-1, 0, field.index("x")]
                    for f, s, l, d in zip(field, scales, labels,
                                          np.rollaxis(v, 2)):
                        name = "%s.%s.%s.png" % (re.sub("\.mat", "",
                                                        path), k, f)
                        plt.imshow(s * d,
                                   vmin=vmin[f] if args.c else None,
                                   vmax=vmax[f] if args.c else None,
                                   extent=(t0, t1, x0, x1),
                                   origin="lower",
                                   aspect="auto")
                        plt.gca().set_ylim(plt.gca().get_ylim()[::-1])
                        t0a = t0 if (args.t0 is None) else 1000 * args.t0
                        t1a = t1 if (args.t1 is None) else 1000 * args.t1
                        plt.axis((t0a, t1a, x0, x1))
                        cb = plt.colorbar(label=l)
                        plt.xlabel("Time (ms)")
                        plt.ylabel("Width (mm)")
                        plt.savefig(name)
                        plt.close()
                        print("view.py: %s" % name)
                elif np.squeeze(v).ndim == 1:
                    name = "%s.%s.png" % (re.sub("\.mat", "", path), k)
                    plt.plot(np.squeeze(v))
                    plt.savefig(name)
                    print("view.py: %s" % name)
                    plt.close()
