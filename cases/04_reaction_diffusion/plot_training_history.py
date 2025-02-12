#!/usr/bin/env python3

import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=str, help="csv file that contains training history")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    epoch = df['epoch'].to_numpy()
    pde_loss = df['pde_loss'].to_numpy()
    data_loss = df['data_loss'].to_numpy()
    loss = df['loss'].to_numpy()

    fig, ax = plt.subplots()
    ax.plot(epoch, pde_loss, label='PDE')
    ax.plot(epoch, data_loss, label='data')
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
