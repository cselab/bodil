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
    omega = np.sqrt(k / m)
    x0 = 0.5
    v0 = 0.2

    seed = 2349873
    num_epochs = 5000
    num_samples = 10000
    lr = 1e-3
    num_data = 20
    sigma_data = 0.1
    beta = 1e4
    rng = np.random.default_rng(seed=seed)

    td, xd = generate_data(num_data, T, omega=omega, x0=x0, v0=v0, rng=rng, sigma=sigma_data)
    xd = torch.from_numpy(xd)

    for nt in [31, 63, 127]:
        t = np.linspace(0, T, nt + 1, endpoint=True)

        xexact = v0/omega * np.sin(omega * t) + x0 * np.cos(omega * t)
        vexact = v0 * np.cos(omega * t) - x0 * omega * np.sin(omega * t)

        dt = t[1] - t[0]

        y = torch.zeros((nt + 1) * 2, requires_grad=True)

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
        x = y[:nt+1]
        v = y[nt+1:]

        H = H.detach().numpy()
        cov = np.linalg.inv(H)

        if False:
            fig, ax = plt.subplots()
            im = ax.imshow(cov, origin='lower', cmap="seismic")
            fig.colorbar(im, ax=ax)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            plt.show()
            plt.close

        with open(f'cov_nt_{nt}.npy', 'wb') as f:
            np.save(f, cov)

        # sample solutions x.
        num_samples = 5000
        samples = np.zeros((len(x) + len(v), num_samples))

        eigvals, eigvecs = np.linalg.eig(H)

        for k in range(num_samples):
            z = rng.normal(0, 1/np.sqrt(eigvals), len(x) + len(v))
            samples[:,k] = y + eigvecs @ z

        xmean = np.mean(samples[:len(x)], axis=1)
        xlo = np.quantile(samples[:len(x)], q=0.05, axis=1)
        xhi = np.quantile(samples[:len(x)], q=0.95, axis=1)

        vmean = np.mean(samples[len(x):], axis=1)
        vlo = np.quantile(samples[len(x):], q=0.05, axis=1)
        vhi = np.quantile(samples[len(x):], q=0.95, axis=1)

        if False:
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
        df.to_csv(f'nt_{nt}.csv', index=False)

    data = {
        't': td,
        'x': xd.detach().numpy()
    }

    df = pd.DataFrame(data)
    df.to_csv('laplace_data.csv', index=False)




if __name__ == '__main__':
    main()
