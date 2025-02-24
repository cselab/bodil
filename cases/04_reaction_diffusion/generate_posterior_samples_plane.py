#!/usr/bin/env python

import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import RBFInterpolator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('csv', type=str, help='csv files that contains losses against parameter x0')
    parser.add_argument('--sigma', type=float, default=0.05, help='data measurements error')
    parser.add_argument('--nsamples', type=int, default=1000, help='number of samples to generate')
    args = parser.parse_args()

    nsamples = args.nsamples
    csv_path = args.csv
    n = 128 # resolution for the interpolated function

    # ODIL loss has term lambda_data * mean(delta u^2), mean over nx * ny data points.
    # equivalent "likelihood" for data would be sum delta u^2 / 2 sigma^2, where sigma is measurements error
    # pseudo-likelihood is exp(-beta * ODLI_loss), so we can compute beta to correspond to a given sigma.
    nx = ny = 64
    lambda_data = 10
    sigma = args.sigma
    beta = nx * ny / (2 * lambda_data * sigma**2)

    df = pd.read_csv(csv_path)
    x0 = df['x0'].to_numpy()
    y0 = df['y0'].to_numpy()
    ODIL_loss = df['ODIL_loss'].to_numpy()

    idx = np.lexsort((y0, x0))

    x0 = x0[idx]
    y0 = y0[idx]
    ODIL_loss = ODIL_loss[idx]

    interp = RBFInterpolator(np.vstack((x0, y0)).T,
                             ODIL_loss)


    xone = np.linspace(x0.min(), x0.max(), n, endpoint=True)
    yone = np.linspace(y0.min(), y0.max(), n, endpoint=True)

    X, Y = np.meshgrid(xone, yone, indexing='ij')
    dx = xone[1] - xone[0]
    dy = yone[1] - yone[0]

    loss = interp(np.vstack((X.flatten(), Y.flatten())).T).reshape((n, n))


    p = np.exp(-beta * loss)
    norm = np.sum(p[:-1,:-1] + p[1:,:-1] + p[1:,1:]) * dx * dy / 4
    p /= norm

    fig, ax = plt.subplots()

    ax.contourf(X, Y, p, levels=100)
    ax.set_xlabel(r'$x_0$')
    ax.set_ylabel(r'$y_0$')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
