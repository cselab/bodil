#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.optim import Adam

from uq_odil.HMC import HMC

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
    omega_ref = np.sqrt(k / m)
    x0 = 0.5
    v0 = 0.2

    seed = 2349873
    num_epochs = 10000
    num_samples = 100000
    lr = 5e-4
    num_data = 20
    sigma_data = 0.1
    beta = 1e4
    rng = np.random.default_rng(seed=seed)

    td, xd = generate_data(num_data, T, omega=omega_ref, x0=x0, v0=v0, rng=rng, sigma=sigma_data)

    nt = 63
    t = np.linspace(0, T, nt + 1, endpoint=True)
    dt = t[1] - t[0]
    y = torch.zeros(1 + (nt + 1) * 2, requires_grad=True)

    td_ids = torch.from_numpy((td / dt).astype(int))
    xd = torch.from_numpy(xd)

    optim = Adam([y], lr=lr)

    def neg_log_posterior(y):
        omega_sq = y[0]
        x = y[1:nt+2]
        v = y[nt+2:]
        dxdt = torch.diff(x) / dt
        dvdt = torch.diff(v) / dt
        xm = (x[:-1] + x[1:]) / 2
        vm = (v[:-1] + v[1:]) / 2

        ode1_res = dxdt - vm
        ode2_res = dvdt + omega_sq * xm
        data_res = x[td_ids] - xd

        loss_PDE = torch.mean(ode1_res**2 + ode2_res**2)
        nlg  = beta * loss_PDE
        nlg -= torch.sum(-data_res**2 / (2 * sigma_data**2)) - num_data/2 * np.log(2 * np.pi * sigma_data**2)
        return nlg

    epochs = list(range(num_epochs))
    losses = []
    for epoch in epochs:
        optim.zero_grad()
        loss = neg_log_posterior(y)
        loss.backward()
        optim.step()
        l = float(loss.detach().cpu().float())
        if epoch % 1000 == 0:
            print(f"epoch {epoch:06d}, loss {l:.6e}")

        losses.append(l)


    hmc = HMC([y], dt=0.0077, L=10, M=1)

    def closure():
        hmc.zero_grad()
        U = neg_log_posterior(y)
        U.backward()
        return U

    samples = []
    num_accepted = 0
    Umap = None
    ymap = None
    for k in tqdm(range(num_samples)):
        y_, H_, U_, accepted = hmc.step(closure)
        samples.append(y_[0].detach().numpy())
        num_accepted += accepted
        if Umap is None or Umap > U_:
            ymap = y_[0].detach().numpy()
            Umap = U_

    print(f"accptance rate: {num_accepted/num_samples}")
    samples = np.array(samples)

    omegasq_samples = samples[:,0]

    x_samples = samples[:,1:nt+2]
    x_map = ymap[1:nt+2]
    x_mean = np.mean(x_samples, axis=0)
    x_lo = np.quantile(x_samples, q=0.05, axis=0)
    x_hi = np.quantile(x_samples, q=0.95, axis=0)

    v_samples = samples[:,nt+2:]
    v_map = ymap[nt+2:]
    v_mean = np.mean(v_samples, axis=0)
    v_lo = np.quantile(v_samples, q=0.05, axis=0)
    v_hi = np.quantile(v_samples, q=0.95, axis=0)

    xexact = v0/omega_ref * np.sin(omega_ref * t) + x0 * np.cos(omega_ref * t)
    vexact = v0 * np.cos(omega_ref * t) - x0 * omega_ref * np.sin(omega_ref * t)

    if 1:
        fig, ax = plt.subplots()
        ax.fill_between(t, x_lo, x_hi, lw=0, color='r', alpha=0.2)
        ax.plot(t, x_mean, '-r')
        ax.plot(t, xexact, '--k')
        ax.plot(td, xd.detach().numpy(), '+k')
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(r"$x$")
        ax.set_xlim(0, T)
        plt.show()

    # estimate covariance
    mean = np.mean(samples, axis=0)
    X = samples - mean[None, :]

    cov = (X.T @ X) / (num_samples - 1)

    fig, ax = plt.subplots()
    #im = ax.imshow(np.linalg.inv(cov), origin='lower', cmap="Reds")
    im = ax.imshow(cov, origin='lower', cmap="seismic")
    fig.colorbar(im, ax=ax)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.show()
    plt.close()

    with open('hmc_cov.npy', 'wb') as f:
        np.save(f, cov)

    data = {
        't': t,
        'xmap': x_map,
        'xmean': x_mean,
        'xexact': xexact,
        'x05': x_lo,
        'x95': x_hi,
        'vmap': v_map,
        'vmean': v_mean,
        'vexact': vexact,
        'v05': v_lo,
        'v95': v_hi
    }

    df = pd.DataFrame(data)
    df.to_csv('hmc_pred.csv', index=False)

    data = {
        't': td,
        'x': xd.detach().numpy()
    }

    df = pd.DataFrame(data)
    df.to_csv('hmc_data.csv', index=False)

    data = {
        'omegasq': omegasq_samples
    }
    print(f"omegasq: mean {np.mean(omegasq_samples)}, std {np.std(omegasq_samples)}")
    df = pd.DataFrame(data)
    df.to_csv('hmc_omega.csv', index=False)


if __name__ == '__main__':
    main()
