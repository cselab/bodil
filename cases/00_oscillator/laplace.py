#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
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
    t = rng.uniform(0, T, num_data)
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
    sigma_ode = 0.5
    rng = np.random.default_rng(seed=seed)

    td, xd = generate_data(num_data, T, omega=omega, x0=x0, v0=v0, rng=rng, sigma=sigma_data)

    nt = 63
    t = np.linspace(0, T, nt + 1, endpoint=True)

    xexact = v0/omega * np.sin(omega * t) + x0 * np.cos(omega * t)

    dt = t[1] - t[0]

    y = torch.zeros((nt + 1, 2), requires_grad=True)

    td_ids = torch.from_numpy((td / dt).astype(int))
    xd = torch.from_numpy(xd)


    optim = Adam([y], lr=lr)

    def neg_log_posterior(y):
        x = y[:,0]
        v = y[:,1]
        dxdt = torch.diff(x) / dt
        dvdt = torch.diff(v) / dt
        xm = (x[:-1] + x[1:]) / 2
        vm = (v[:-1] + v[1:]) / 2

        ode1_res = dxdt - vm
        ode2_res = dvdt + omega * xm
        data_res = x[td_ids] - xd

        log_like  = torch.sum(-ode1_res**2 / (2 * sigma_ode**2)) - nt/2 * np.log(2 * np.pi * sigma_ode**2)
        log_like += torch.sum(-ode2_res**2 / (2 * sigma_ode**2)) - nt/2 * np.log(2 * np.pi * sigma_ode**2)
        log_like += torch.sum(-data_res**2 / (2 * sigma_data**2)) - num_data/2 * np.log(2 * np.pi * sigma_data**2)
        log_like += -(x[0] - x0)**2 / (2 * sigma_ode**2) - 1/2 * np.log(2 * np.pi * sigma_ode**2)

        return -log_like

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

    x = y[:,0].detach().numpy()

    H = H.detach().numpy()
    Hx = H[:,0,:,0]

    fig, ax = plt.subplots()
    ax.imshow(Hx, origin='lower')
    ax.set_title(r"Hesian of negative log likelihood along $x$")
    plt.show()
    plt.close

    # sample solutions x.
    num_samples = 5000
    samples = np.zeros((len(x), num_samples))

    eigvals, eigvecs = np.linalg.eig(Hx)

    for k in range(num_samples):
        z = rng.normal(0, 1/np.sqrt(eigvals), len(x))
        samples[:,k] = x + eigvecs @ z

    xmean = np.mean(samples, axis=1)
    xlo = np.quantile(samples, q=0.05, axis=1)
    xhi = np.quantile(samples, q=0.95, axis=1)

    fig, ax = plt.subplots()
    ax.fill_between(t, xlo, xhi, lw=0, alpha=0.2, color='r', label='5-95% quantiles of posterior')
    ax.plot(t, x, '-r', label='MAP')
    ax.plot(t, xexact, '--k', label='exact')
    ax.plot(td, xd.detach().numpy(), '+k', label='data')
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$x$")
    ax.set_xlim(0, T)
    ax.set_ylim(-1.5, 1.5)
    ax.legend(frameon=False)
    plt.show()







if __name__ == '__main__':
    main()
