#!/usr/bin/env python

import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('csv', type=str, help='csv files that contains losses against parameter x0')
    parser.add_argument('--beta', type=float, default=3e4, help='inverse "temperature"')
    args = parser.parse_args()

    csv_path = args.csv
    beta = args.beta

    df = pd.read_csv(csv_path)
    x0 = df['x0'].to_numpy()
    ODIL_loss = df['ODIL_loss'].to_numpy()

    dx = x0[1] - x0[0]

    p = np.exp(-beta * ODIL_loss)
    p /= np.sum(p * dx)

    fig, ax = plt.subplots()
    ax.bar(x0, p, width=dx)
    ax.set_xlabel(r'$x_0$')
    ax.set_ylabel(r'$p(x_0 | D)$')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, None)
    plt.show()


if __name__ == '__main__':
    main()
