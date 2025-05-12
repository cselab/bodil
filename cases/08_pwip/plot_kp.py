#!/usr/bin/env python

import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_odil', type=str, help='path to csv file containing kp for ODIL')
    parser.add_argument('data_pinn', type=str, help='path to csv file containing kp for PINNs')
    args = parser.parse_args()

    df = pd.read_csv(args.data_odil)
    xm = df['x'].to_numpy()
    kp_odil = df['kp'].to_numpy()

    data_pinn = np.loadtxt(args.data_pinn)
    print(data_pinn.shape)
    x, kp_pinn = data_pinn.T
    kp_pinn = (kp_pinn[1:] + kp_pinn[:-1]) / 2

    fig, ax = plt.subplots()
    ax.plot(xm, kp_odil / 2, '+k')
    ax.plot(xm, kp_pinn, 'ob')
    #ax.plot(kp_odil, kp_pinn, 'ok')
    ax.set_xlabel(r"$x$ (mm)")
    ax.set_ylabel(r"$k_p$ (m$^3$ s$^2$ / kg)")
    plt.tight_layout()
    plt.show()
    plt.close()

if __name__ == '__main__':
    main()
