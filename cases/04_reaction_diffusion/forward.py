#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import os

from random_field import generate_random_field

def main():
    out_dir = "out_forward"
    os.makedirs(out_dir, exist_ok=True)
    Dw = 0.005
    Dg = 0.1
    rho = 8
    L = 1.0
    nx = 64
    ny = 64
    tend = 1.0
    t_every = 0.01
    rng = np.random.default_rng(seed=123456)

    x = np.linspace(0, L, nx)
    y = np.linspace(0, L, ny)

    dx = x[1] - x[0]
    dy = y[1] - y[0]

    diff_field = generate_random_field(nx, ny, smoothness=nx/8, rng=rng)
    diff_field = np.where(diff_field > 0, Dw, Dg)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(diff_field, origin='lower', extent=[0, L, 0, L], vmin=Dw, vmax=Dg)
    plt.savefig(os.path.join(out_dir, "diff_field.png"))
    plt.close(fig)

    with open(os.path.join(out_dir, "diff_field.npy"), "wb") as f:
        np.save(f, diff_field)

    X, Y = np.meshgrid(x, y)

    # initial conditions
    u = np.where((X-2*L/3)**2 + (Y-L/3)**2 < (L/32)**2,
                 1.0, 0.0)

    dt = 0.25 * min([dx, dy])**2 / max([Dg, Dw])
    t = 0

    next_tdump = 0
    dump_id = 0
    iteration = 0

    while t < tend:
        Dm0 = (diff_field + np.roll(diff_field, shift=+1, axis=1)) / 2
        Dp0 = (diff_field + np.roll(diff_field, shift=-1, axis=1)) / 2
        D0m = (diff_field + np.roll(diff_field, shift=+1, axis=0)) / 2
        D0p = (diff_field + np.roll(diff_field, shift=-1, axis=0)) / 2

        um0 = np.roll(u, shift=+1, axis=1)
        up0 = np.roll(u, shift=-1, axis=1)
        u0m = np.roll(u, shift=+1, axis=0)
        u0p = np.roll(u, shift=-1, axis=0)

        rhs  = (Dp0 * (up0 - u) - Dm0 * (u - um0)) / dx**2
        rhs += (D0p * (u0p - u) - D0m * (u - u0m)) / dy**2
        rhs += rho * u * (1 - u)

        dt = min([tend - t, dt])

        u += dt * rhs
        t += dt
        iteration += 1

        if t >= next_tdump:
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(u, origin='lower', extent=[0, L, 0, L], vmin=0, vmax=1)
            plt.savefig(os.path.join(out_dir, f"u-{dump_id:06d}.png"))
            plt.close(fig)
            if dump_id % 10 == 0:
                print(f"iteration {iteration:06d} (time {t:.4f})")
            next_tdump += t_every
            dump_id += 1

    with open(os.path.join(out_dir, "u_final.npy"), "wb") as f:
        np.save(f, u)


if __name__ == '__main__':
    main()
