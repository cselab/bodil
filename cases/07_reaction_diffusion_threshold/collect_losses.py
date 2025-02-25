#!/usr/bin/env python

import argparse
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
    parser.add_argument('--show-plot', action='store_true', default=False, help='show plot')
    args = parser.parse_args()

    paths = args.out_inverse_paths

    x0 = []
    y0 = []
    losses = []
    for path in paths:
        try:
            p = read_params(path)
            l = compute_loss(path)
        except:
            pass
        else:
            x0.append(p['x0'])
            y0.append(p['y0'])
            losses.append(l)

    x0 = np.array(x0)
    y0 = np.array(y0)
    losses = np.array(losses)

    data = {
        'x0': x0,
        'y0': y0,
        'ODIL_loss': losses
    }
    pd.DataFrame(data).to_csv(args.out_csv, index=False)

    if args.show_plot:
        import matplotlib
        import matplotlib.pyplot as plt

        if len(np.unique(y0)) == 1:
            fig, ax = plt.subplots()
            ax.plot(x0, losses, '-o')
            ax.set_xlabel(r'$x_0$')
            ax.set_ylabel('ODIL loss')
            ax.set_yscale('log')
            plt.tight_layout()
            plt.show()
            plt.close()
        else:
            fig, ax = plt.subplots()
            sc = ax.scatter(x0, y0, c=losses,
                            norm=matplotlib.colors.LogNorm())
            plt.colorbar(sc, ax=ax)
            ax.set_xlabel(r'$x_0$')
            ax.set_ylabel(r'$y_0$')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect('equal')
            plt.tight_layout()
            plt.show()
            plt.close()





if __name__ == '__main__':
    main()
