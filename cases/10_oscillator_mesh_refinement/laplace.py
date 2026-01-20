#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

import torch
from torch.optim import Adam

def generate_data(num_data, T, omega, x0, v0, rng, sigma):
    """
    Parameters:
        num_data: number of samples to generate
        T: time horizon
        omega: frequency of the oscillator
        x0: initial position
        v0: initial velocity
        rng: numpy random number generator
        sigma: noise level
    """
    t = rng.uniform(0, T/2, num_data)
    x = v0 / omega * np.sin(omega * t) + x0 * np.cos(omega * t) + rng.normal(0, sigma, num_data)
    return t, x


def main():
    # a particle of mass m with position x and velocity v, in harmonic force -k x.

    T = 20.0
    k = 1.0
    m = 1.0
    omega = np.sqrt(k / m)
    x0 = 0.5
    v0 = 0.2

    seed = 2349873
    num_epochs = 50000
    lr = 1e-4
    num_data = 20
    sigma_data = 0.1
    beta = 1e4
    rng = np.random.default_rng(seed=seed)

    td, xd = generate_data(num_data, T, omega=omega, x0=x0, v0=v0, rng=rng, sigma=sigma_data)
    xd = torch.from_numpy(xd)

    nts = []
    mus = []
    sigmas = []

    fig, ax = plt.subplots()

    t_prev = None
    y_prev = None

    for nt in [15, 31, 63, 127, 255, 511, 1023, 2047, 4095, 8191]:
        t = np.linspace(0, T, nt + 1, endpoint=True)

        if nt > 511:
            lr = 5e-5

        xexact = v0/omega * np.sin(omega * t) + x0 * np.cos(omega * t)
        vexact = v0 * np.cos(omega * t) - x0 * omega * np.sin(omega * t)

        dt = t[1] - t[0]

        if y_prev is None:
            y_init = np.zeros((nt + 1) * 2, dtype=np.float64)
        else:
            x_prev = y_prev[:len(t_prev)]
            v_prev = y_prev[len(t_prev):]
            x_init = np.interp(t, t_prev, x_prev)
            v_init = np.interp(t, t_prev, v_prev)
            y_init = np.concatenate([x_init, v_init]).astype(np.float64)

        y = torch.tensor(y_init, requires_grad=True)

        td_ids = torch.from_numpy((td / dt).astype(int))


        optim = Adam([y], lr=lr)

        def neg_log_posterior(y):
            x = y[:nt+1]
            v = y[nt+1:]
            dxdt = torch.diff(x) / dt
            dvdt = torch.diff(v) / dt
            xm = (x[:-1] + x[1:]) / 2
            vm = (v[:-1] + v[1:]) / 2

            ode1_res = dxdt - vm
            ode2_res = dvdt + omega * xm
            data_res = x[td_ids] - xd

            loss_PDE = torch.mean(ode1_res**2 + ode2_res**2)
            nlg  = beta * loss_PDE
            nlg -= torch.sum(-data_res**2 / (2 * sigma_data**2)) - num_data/2 * np.log(2 * np.pi * sigma_data**2)
            return nlg

        epochs = list(range(num_epochs+1))
        losses = []
        for epoch in epochs:
            optim.zero_grad()
            loss = neg_log_posterior(y)
            loss.backward()
            optim.step()
            l = float(loss.detach().cpu().float())
            if epoch % 10000 == 0:
                print(f"epoch {epoch:06d}, loss {l:.6e}")

            losses.append(l)

        H = torch.autograd.functional.hessian(neg_log_posterior, y, create_graph=True)

        y = y.detach().numpy()
        x = y[:nt+1]
        v = y[nt+1:]

        H = H.detach().numpy()
        cov = np.linalg.inv(H)

        # with open(f'cov_nt_{nt}.npy', 'wb') as f:
        #     np.save(f, cov)

        i = nt
        mu = y[i]
        sigma = np.sqrt(cov[i, i])

        nts.append(nt+1)
        mus.append(mu)
        sigmas.append(sigma)

        xx = np.linspace(0, 1, 500)
        px = norm.pdf(xx, mu, sigma)

        ax.plot(xx, px, label=f'nt={nt+1}')

        t_prev = t.copy()
        y_prev = y.copy()


    ax.legend()
    plt.show()

    data = {
        'Nt': nts,
        'mu': mus,
        'sigma': sigmas
    }

    df = pd.DataFrame(data)
    df.to_csv('pred_x_20.csv', index=False)


if __name__ == '__main__':
    main()
