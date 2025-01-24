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
    k0 = 1.0
    m0 = 1.0
    omega0 = np.sqrt(k0 / m0)
    x0 = 0.5
    v0 = 0.2

    seed = 2349873
    num_epochs = 2000
    lr = 1e-3
    num_data = 20
    lambda_data = 1
    sigma_data = 0.1
    rng = np.random.default_rng(seed=seed)

    td, xd = generate_data(num_data, T, omega=omega0, x0=x0, v0=v0, rng=rng, sigma=sigma_data)

    nt = 32

    t = np.linspace(0, T, nt + 1, endpoint=True)
    dt = t[1] - t[0]
    td_ids = (td / dt).astype(int)
    td_ids_ = torch.from_numpy(td_ids)
    xd_ = torch.from_numpy(xd)

    def odil(omega):
        y = torch.zeros((nt + 1, 2), requires_grad=True)
        optim = Adam([y], lr=lr)

        def compute_loss():
            x = y[:,0]
            v = y[:,1]
            dxdt = torch.diff(x) / dt
            dvdt = torch.diff(v) / dt
            xm = (x[:-1] + x[1:]) / 2
            vm = (v[:-1] + v[1:]) / 2

            ode1_res = dxdt - vm
            ode2_res = dvdt + omega * xm
            data_res = x[td_ids_] - xd_

            loss = torch.mean(ode1_res**2) + torch.mean(ode2_res**2)
            loss += lambda_data * torch.mean(data_res**2)
            return loss

        for epoch in range(num_epochs):
            optim.zero_grad()
            loss = compute_loss()
            loss.backward()
            optim.step()
            l = float(loss.detach().cpu().float())

        return y.detach().numpy()


    def log_posterior(omega, sigma):
        y = odil(omega)
        x = y[:,0]

        data_res = x[td_ids] - xd

        log_like = np.sum(-data_res**2 / (2 * sigma**2)) - num_data / 2 * np.log(2 * np.pi * sigma**2)

        return log_like


    omegas = np.linspace(0.0, 2.0, 100)
    p = []
    sigma = sigma_data
    for w in omegas:
        p.append(log_posterior(w, sigma))

    dw = omegas[1] - omegas[0]
    p = np.array(p)
    p = np.exp(p)
    p /= dw * np.sum(p)

    fig, ax = plt.subplots()
    ax.plot(omegas, p)
    ax.set_xlabel(r'$\omega$')
    ax.set_ylabel(r'$p(\omega | D)$')
    ax.set_xlim(0, 2)
    ax.set_ylim(0, None)
    plt.show()






if __name__ == '__main__':
    main()
