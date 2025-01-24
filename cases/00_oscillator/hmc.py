#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
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
    t = rng.uniform(0, T/4, num_data)
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
    num_data = 10
    sigma_data = 0.1
    sigma_ode = 0.1
    rng = np.random.default_rng(seed=seed)

    td, xd = generate_data(num_data, T, omega=omega, x0=x0, v0=v0, rng=rng, sigma=sigma_data)

    nt = 63
    t = np.linspace(0, T, nt + 1, endpoint=True)
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


    hmc = HMC([y], dt=0.0116, L=10, M=1)

    def closure():
        hmc.zero_grad()
        U = neg_log_posterior(y)
        U.backward()
        return U

    samples = []
    num_accepted = 0
    for k in range(num_samples):
        y_, H_, accepted = hmc.step(closure)
        samples.append(y_[0].detach().numpy())
        num_accepted += accepted

    print(f"accptance rate: {num_accepted/num_samples}")
    samples = np.array(samples)

    x_samples = samples[:, :, 0]
    x_mean = np.mean(x_samples, axis=0)
    x_lo = np.quantile(x_samples, q=0.1, axis=0)
    x_hi = np.quantile(x_samples, q=0.9, axis=0)

    xexact = v0/omega * np.sin(omega * t) + x0 * np.cos(omega * t)

    fig, ax = plt.subplots()
    ax.fill_between(t, x_lo, x_hi, lw=0, color='r', alpha=0.2)
    ax.plot(t, x_mean, '-r')
    ax.plot(t, xexact, '--k')
    ax.plot(td, xd.detach().numpy(), '+k')
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$x$")
    ax.set_xlim(0, T)
    plt.show()






if __name__ == '__main__':
    main()
