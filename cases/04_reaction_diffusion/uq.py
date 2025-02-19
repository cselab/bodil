#!/usr/bin/env python

import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('csv', type=str, help='csv files that contains losses against parameter x0')
    parser.add_argument('--beta', type=float, default=2e4, help='inverse "temperature"')
    args = parser.parse_args()

    csv_path = args.csv
    beta = args.beta

    df = pd.read_csv(csv_path)
    df.sort_values(by='x0', inplace=True)
    x0 = df['x0'].to_numpy()
    ODIL_loss = df['ODIL_loss'].to_numpy()

    f = CubicSpline(x0, ODIL_loss, bc_type='natural')
    x = np.linspace(x0.min(), x0.max(), 1024)
    y = f(x)

    p = np.exp(-beta * y)
    norm = np.sum((p[1:] + p[:-1]) / 2 * np.diff(x))
    p /= norm

    fig, ax = plt.subplots()
    ax.plot(x, p)
    ax.set_xlabel(r'$x_0$')
    ax.set_ylabel(r'$p(x_0 | D)$')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, None)
    plt.show()


if __name__ == '__main__':
    main()
