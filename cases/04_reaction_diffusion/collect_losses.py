#!/usr/bin/env python

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

def read_params(path):
    d = path.split('/')[-1].split('_')
    params = {}
    for i in reversed(range(len(d))):
        try:
            val = float(d[i])
            key = d[i-1]
            params[key] = val
        except:
            pass
    return params

def compute_loss(path):
    df = pd.read_csv(os.path.join(path, 'train_history.csv'))
    loss = df['loss'].to_numpy()
    n = len(loss) // 10
    return np.mean(loss[-n:])
    #return loss.min()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('out_inverse_paths', type=str, nargs='+', help='paths of output directories of inverse.py')
    parser.add_argument('--out-csv', type=str, default='losses.csv', help='output csv')
    args = parser.parse_args()

    paths = args.out_inverse_paths

    x0 = []
    losses = []
    for path in paths:
        try:
            p = read_params(path)
            l = compute_loss(path)
        except:
            pass
        else:
            x0.append(p['x0'])
            losses.append(l)

    x0 = np.array(x0)
    losses = np.array(losses)
    idx = np.argsort(x0)
    x0 = x0[idx]
    losses = losses[idx]

    data = {
        'x0': x0,
        'ODIL_loss': losses
    }
    pd.DataFrame(data).to_csv(args.out_csv, index=False)

    fig, ax = plt.subplots()
    ax.plot(x0, losses, '-o')
    ax.set_xlabel(r'$x_0$')
    ax.set_ylabel('ODIL loss')
    ax.set_yscale('log')
    plt.tight_layout()
    plt.show()




if __name__ == '__main__':
    main()
