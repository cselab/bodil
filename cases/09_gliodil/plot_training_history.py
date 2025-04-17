#!/usr/bin/env python3

import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def window_average(signal, w):
    ret = np.cumsum(signal)
    ret[w:] = ret[w:] - ret[:-w]
    return ret[w-1:] / w

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=str, help="csv file that contains training history")
    parser.add_argument("--window-avg", type=int, default=1, help="window average width")
    args = parser.parse_args()

    w = args.window_avg

    df = pd.read_csv(args.csv_path)
    epoch = df['epoch'].to_numpy()
    pde_loss = df['pde_loss'].to_numpy()
    data_loss = df['data_loss'].to_numpy()
    ic_loss = df['ic_loss'].to_numpy()
    loss = df['loss'].to_numpy()

    if w > 1:
        pde_loss_ = window_average(pde_loss, w)
        data_loss_ = window_average(data_loss, w)
        ic_loss_ = window_average(ic_loss, w)
        loss_ = window_average(loss, w)
        epoch_ = epoch[:-w+1]

    fig, ax = plt.subplots()
    if w > 1:
        ax.plot(epoch_, pde_loss_, label='PDE')
        ax.plot(epoch_, data_loss_, label='data')
        ax.plot(epoch_, ic_loss_, label='IC')
        ax.plot(epoch_, loss_, label='total')
    else:
        ax.plot(epoch, pde_loss, label='PDE')
        ax.plot(epoch, data_loss, label='data')
        ax.plot(epoch, ic_loss, label='IC')
        ax.plot(epoch, loss, label='total')
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax.legend()
    ax.set_yscale('log')
    ax.set_ylim(np.min([pde_loss[-1], data_loss[-1]])/2, None)
    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    main()
