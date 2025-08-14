#!/usr/bin/env python

import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hmc', type=str, default="hmc_omega.csv", help="omega samples from HMC")
    parser.add_argument('--lap', type=str, default="laplace_omega.csv", help="omega moments from Laplace")
    parser.add_argument('--profile', type=str, default="profile_omega.csv", help="profile omega")
    args = parser.parse_args()

    df = pd.read_csv(args.hmc)
    omegas = df['omegasq'].to_numpy()

    df = pd.read_csv(args.lap)
    mul = df['omegasq_mean'][0]
    stdl = df['omegasq_std'][0]

    df = pd.read_csv(args.profile)
    om_pro = df['omegasq'].to_numpy()
    pom_pro = df['pomegasq'].to_numpy()

    om = np.linspace(mul - 5 * stdl, mul + 5 * stdl, 1000)
    pom = np.exp(-(om - mul)**2 / (2 * stdl**2)) / np.sqrt(2 * np.pi * stdl**2)

    fig, ax = plt.subplots()
    ax.hist(omegas, bins=50, density=True, label='HMC', color='C0')
    ax.plot(om, pom, '-', label='Laplace', color='C1')
    ax.plot(om_pro, pom_pro, '--', label='Profile likelihood', color='C2')
    ax.set_xlabel(r"$\omega^2$")
    ax.set_ylabel(r"$P(\omega^2 | D)$")
    ax.set_xlim(0.7, 1.3)
    ax.set_ylim(0, None)
    ax.legend()

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
