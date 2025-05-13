#!/usr/bin/env python3
import sys
import scipy
import numpy as np
import itertools

skip = set(["__header__", "__version__", "__globals__"])
field = "A", "u", "P", "t", "x"

for path in itertools.islice(sys.argv, 1, None):
    print("%s" % path)
    data = scipy.io.loadmat(path)
    for k, v in data.items():
        if k not in skip:
            if np.size(v):
                print("%15s %15s" % (k, np.shape(v)))
                if v.ndim == 3:
                    mi = np.min(v, axis=(0, 1))
                    ma = np.max(v, axis=(0, 1))
                    for f, a, b in zip(field, mi, ma):
                        print("           [%s]: % 8.2e % 8.2e" % (f, a, b))
                else:
                    mi = np.min(v)
                    ma = np.max(v)
                    print("              : % 8.2e % 8.2e" % (mi, ma))
