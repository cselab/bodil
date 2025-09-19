#!/usr/bin/env python

import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hmc', type=str, default="hmc_omega.csv", help="omega samples from HMC")
    args = parser.parse_args()

    df = pd.read_csv(args.hmc)
    omegas = df['omegasq'].to_numpy()[:10000]
    n = np.arange(len(omegas))

    fig, ax = plt.subplots()
    ax.plot(n, omegas, '.')
    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"$\omega^2_k$")
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
