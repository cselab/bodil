#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
    omega_ref = np.sqrt(k / m)
    x0 = 0.5
    v0 = 0.2

    seed = 2349873
    num_epochs = 10000
    num_samples = 10000
    lr = 5e-4
    num_data = 20
    sigma_data = 0.1
    beta = 1e4
    rng = np.random.default_rng(seed=seed)

    td, xd = generate_data(num_data, T, omega=omega_ref, x0=x0, v0=v0, rng=rng, sigma=sigma_data)

    nt = 63
    t = np.linspace(0, T, nt + 1, endpoint=True)

    xexact = v0/omega_ref * np.sin(omega_ref * t) + x0 * np.cos(omega_ref * t)
    vexact = v0 * np.cos(omega_ref * t) - x0 * omega_ref * np.sin(omega_ref * t)

    dt = t[1] - t[0]

    y = torch.zeros(1 + (nt + 1) * 2, requires_grad=True)

    td_ids = torch.from_numpy((td / dt).astype(int))
    xd = torch.from_numpy(xd)


    optim = Adam([y], lr=lr)

    def neg_log_posterior(y):
        omegasq = y[0]
        x = y[1:nt+2]
        v = y[nt+2:]
        dxdt = torch.diff(x) / dt
        dvdt = torch.diff(v) / dt
        xm = (x[:-1] + x[1:]) / 2
        vm = (v[:-1] + v[1:]) / 2

        ode1_res = dxdt - vm
        ode2_res = dvdt + omegasq * xm
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

    H = torch.autograd.functional.hessian(neg_log_posterior, y, create_graph=True)

    y = y.detach().numpy()
    omegasq = y[0]
    x = y[1:nt+2]
    v = y[nt+2:]

    H = H.detach().numpy()

    cov = np.linalg.inv(H)
    fig, ax = plt.subplots()
    im = ax.imshow(cov, origin='lower', cmap="seismic")
    fig.colorbar(im, ax=ax)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.show()
    plt.close

    with open('laplace_cov.npy', 'wb') as f:
        np.save(f, cov)

    # sample solutions x.
    num_samples = 5000
    samples = np.zeros((1 + len(x) + len(v), num_samples))

    eigvals, eigvecs = np.linalg.eig(H)

    for k in range(num_samples):
        z = rng.normal(0, 1/np.sqrt(eigvals), 1 + len(x) + len(v))
        samples[:,k] = y + eigvecs @ z

    omegasq_mean = np.mean(samples[0])
    omegasq_lo = np.quantile(samples[0], q=0.05)
    omegasq_hi = np.quantile(samples[0], q=0.95)

    xmean = np.mean(samples[1:1+len(x)], axis=1)
    xlo = np.quantile(samples[1:1+len(x)], q=0.05, axis=1)
    xhi = np.quantile(samples[1:1+len(x)], q=0.95, axis=1)

    vmean = np.mean(samples[1+len(x):], axis=1)
    vlo = np.quantile(samples[1+len(x):], q=0.05, axis=1)
    vhi = np.quantile(samples[1+len(x):], q=0.95, axis=1)

    fig, axes = plt.subplots(ncols=2, figsize=(9.6,4.8))
    ax = axes[0]
    ax.fill_between(t, xlo, xhi, lw=0, alpha=0.2, color='r', label='5-95% quantiles of posterior')
    ax.plot(t, x, '-r', label='MAP')
    ax.plot(t, xexact, '--k', label='exact')
    ax.plot(td, xd.detach().numpy(), '+k', label='data')
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$x$")
    ax.set_xlim(0, T)
    ax.set_ylim(-1.5, 1.5)
    ax.legend(frameon=False)

    ax = axes[1]
    ax.fill_between(t, vlo, vhi, lw=0, alpha=0.2, color='r')
    ax.plot(t, v, '-r')
    ax.plot(t, vexact, '--k')
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$v$")
    ax.set_xlim(0, T)
    ax.set_ylim(-1.5, 1.5)

    plt.tight_layout()
    plt.show()

    data = {
        't': t,
        'xmap': x,
        'xmean': x,
        'xexact': xexact,
        'x05': xlo,
        'x95': xhi,
        'vmap': v,
        'vmean': v,
        'vexact': vexact,
        'v05': vlo,
        'v95': vhi
    }

    df = pd.DataFrame(data)
    df.to_csv('laplace_pred.csv', index=False)

    data = {
        't': td,
        'x': xd.detach().numpy()
    }

    df = pd.DataFrame(data)
    df.to_csv('laplace_data.csv', index=False)

    data = {
        'omegasq_mean': [omegasq.item()],
        'omegasq_std': [np.sqrt(cov[0,0])]
    }

    df = pd.DataFrame(data)
    df.to_csv('laplace_omega.csv', index=False)

if __name__ == '__main__':
    main()
