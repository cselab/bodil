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


def solve_ODIL(omega_sq,
               beta, sigma_data,
               td_ids, xd, dt, nt,
               num_epochs=10000,
               lr=5e-4):

    y = torch.zeros((nt + 1) * 2, requires_grad=True)
    optim = Adam([y], lr=lr)

    def ODIL_loss(y):
        x = y[:nt+1]
        v = y[nt+1:]
        dxdt = torch.diff(x) / dt
        dvdt = torch.diff(v) / dt
        xm = (x[:-1] + x[1:]) / 2
        vm = (v[:-1] + v[1:]) / 2

        ode1_res = dxdt - vm
        ode2_res = dvdt + omega_sq * xm
        data_res = x[td_ids] - xd

        loss_PDE = torch.mean(ode1_res**2 + ode2_res**2)
        nlg  = beta * loss_PDE
        nlg += torch.sum(data_res**2 / (2 * sigma_data**2))
        return nlg

    losses = []
    for epoch in range(num_epochs + 1):
        optim.zero_grad()
        loss = ODIL_loss(y)
        loss.backward()
        optim.step()
        l = float(loss.detach().cpu().float())
        #if epoch % 5000 == 0:
        #    print(f"epoch {epoch:06d}, loss {l:.6e}")

        if epoch > num_epochs * 0.9:
            losses.append(l)

    H = torch.autograd.functional.hessian(ODIL_loss, y, create_graph=True)
    H = H.detach().numpy()
    y = y.detach().numpy()
    mean_loss = np.mean(losses)
    return mean_loss, y, H


def main():
    T = 20.0
    omega_ref = 1
    x0 = 0.5
    v0 = 0.2

    seed = 2349873
    num_data = 20
    sigma_data = 0.1
    beta = 1e4
    rng = np.random.default_rng(seed=seed)

    td, xd = generate_data(num_data, T, omega=omega_ref, x0=x0, v0=v0, rng=rng, sigma=sigma_data)

    nt = 63
    t = np.linspace(0, T, nt + 1, endpoint=True)

    dt = t[1] - t[0]
    td_ids = torch.from_numpy((td / dt).astype(int))
    xd = torch.from_numpy(xd)

     # constant factor for numerical stability of det computation;
     # inprinciple one should then multiply the determinant by Hfactor**len(H), but
     # this does not matter in final result since it is the same for everybody and
     # we rescale the prob in the end.
    Hfactor = 3200

    omegasqs = np.linspace(0.7, 1.3, 50)
    losses = []
    detHs = []
    for omegasq in omegasqs:
        loss, y, H = solve_ODIL(omegasq, beta=beta, sigma_data=sigma_data,
                                dt=dt, nt=nt, xd=xd, td_ids=td_ids)
        detH = np.linalg.det(H / Hfactor)
        print(f"omegasq {omegasq:.2e}, loss {loss:.4e}, det(H) {detH:.4e}")
        losses.append(loss)
        detHs.append(detH)

    losses = np.array(losses)
    detHs = np.array(detHs)
    pomegasq = np.exp(-losses) / np.sqrt(detHs)

    norm = np.sum((pomegasq[1:] + pomegasq[:-1]) / 2 * np.diff(omegasqs))
    pomegasq /= norm

    data = {
        'omegasq': omegasqs,
        'pomegasq': pomegasq
    }
    df = pd.DataFrame(data)
    df.to_csv('exact_omega.csv', index=False)

    fig, ax = plt.subplots()
    ax.plot(omegasqs, pomegasq)
    ax.set_ylim(0, None)
    ax.set_xlim(0.7, 1.3)
    ax.set_xlabel(r"$\omega$")
    plt.show()


if __name__ == '__main__':
    main()
