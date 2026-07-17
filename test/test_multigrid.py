#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import torch

from bodil.multigrid import MultigridField

def main():
    Lx = 2 * np.pi # periodic direction
    Ly = 1 # non periodic
    nx = 8
    ny = 9

    x = np.linspace(0., Lx, nx, endpoint=False)
    y = np.linspace(0., Ly, ny, endpoint=True)
    X, Y = np.meshgrid(x, y, indexing='ij')
    u = np.sin(X) * (0.1 * Y**2)
    u = torch.from_numpy(u)

    mg = MultigridField(u, loc='pn', depth=3)

    u_ = mg.get()

    print(f"reconstruction error: {torch.mean((u - u_)**2).item()}")



if __name__ == '__main__':
    main()
