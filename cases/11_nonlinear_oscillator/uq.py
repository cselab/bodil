#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def generate_data(nt, num_data, T, m, k1, k2, x0, v0, rng, sigma):
    """
    Parameters:
        nt: number of time intervals
        num_data: number of samples to generate
        T: time horizon
        k1, k2: stiffness (Duffing type oscillator)
        m: mass of the oscillator
        x0: initial position
        v0: initial velocity
        rng: numpy random number generator
        sigma: noise level
    """
    dt = T / nt
    t = 0
    xs = [x0]
    vs = [v0]
    ts = [t]

    x, v = x0, v0
    for i in range(nt):
        # Leapfrog
        F = -(k1 * x + k2 * x**3)
        v += dt/2 * F/m
        x += dt * v
        F = -(k1 * x + k2 * x**3)
        v += dt/2 * F/m
        t += dt

        ts.append(t)
        xs.append(x)
        vs.append(v)

    xth = np.array(xs)
    vth = np.array(vs)
    tth = np.array(ts)

    # measurements
    t_meas = rng.uniform(0.0, T/2, num_data)

    x_true = np.interp(t_meas, tth, xth)
    x_meas = x_true + sigma * rng.standard_normal(num_data)

    return t_meas, x_meas, tth, xth, vth


def main():
    # a particle of mass m with position x and velocity v, in harmonic force -k x.

    T = 20.0
    k1 = 1
    k2 = 10
    m = 15.0
    x0 = 2
    v0 = 0.0

    seed = 2349873
    num_samples = 100000
    burnin = 1000
    num_data = 50
    sigma_data = 0.1
    rng = np.random.default_rng(seed=seed)

    td, xd, texact, xexact, vexact = generate_data(
        nt=10000, num_data=num_data,
        T=T, m=m, k1=k1, k2=k2,
        x0=x0, v0=v0,
        rng=rng, sigma=sigma_data)

    nt = 255
    t = np.linspace(0, T, nt + 1, endpoint=True)
    dt = t[1] - t[0]

    td_ids = (td / dt).astype(int)

    def forward(x0, v0, omega):
        t, x, v = 0, x0, v0
        ts = [0]
        xs = [x0]
        vs = [v0]

        for i in range(nt):
            v -= dt/2 * omega**2 * x
            x += dt * v
            v -= dt/2 * omega**2 * x
            t += dt
            xs.append(x)
            vs.append(v)
            ts.append(t)

        return np.array(ts), np.array(xs), np.array(vs)

    def log_posterior(theta):
        x0, v0, omega = theta

        t, x, v = forward(x0, v0, omega)

        data_res = x[td_ids] - xd
        return - np.sum(data_res**2) / (2 * sigma_data**2)

    theta = np.array(
        [2.0, 0.0, 1.0] # x0, v0, omega
    )
    logp = log_posterior(theta)

    accepted = 0
    samples = []
    for i in range(num_samples + burnin):
        thetap = rng.normal(theta, 0.002)
        logpp = log_posterior(thetap)
        loga = logpp - logp
        if np.log(rng.uniform()) < loga:
            if i >= burnin:
                accepted += 1
            theta = thetap
            logp = logpp
        if i >= burnin:
            samples.append(theta.copy())

    print(f"acceptance rate: {accepted / num_samples}")

    samples = np.array(samples)

    all_x = np.empty((nt+1, num_samples))
    all_v = np.empty((nt+1, num_samples))

    for i in range(num_samples):
        x0, v0, omega = samples[i]
        t, x, v = forward(x0, v0, omega)
        all_x[:,i] = x
        all_v[:,i] = v

    xmean = np.mean(all_x, axis=1)
    xlo = np.quantile(all_x, q=0.05, axis=1)
    xhi = np.quantile(all_x, q=0.95, axis=1)

    vmean = np.mean(all_v, axis=1)
    vlo = np.quantile(all_v, q=0.05, axis=1)
    vhi = np.quantile(all_v, q=0.95, axis=1)

    fig, axes = plt.subplots(ncols=2, figsize=(9.6,4.8))
    ax = axes[0]
    ax.fill_between(t, xlo, xhi, lw=0, alpha=0.2, color='r', label='5-95% quantiles of posterior')
    ax.plot(t, x, '-r', label='MAP')
    ax.plot(texact, xexact, '--k', label='exact')
    ax.plot(td, xd, '+k', label='data')
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$x$")
    ax.set_xlim(0, T)
    #ax.set_ylim(-1.5, 1.5)
    ax.legend(frameon=False)

    ax = axes[1]
    ax.fill_between(t, vlo, vhi, lw=0, alpha=0.2, color='r')
    ax.plot(t, v, '-r')
    ax.plot(texact, vexact, '--k')
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$v$")
    ax.set_xlim(0, T)
    #ax.set_ylim(-1.5, 1.5)

    plt.tight_layout()
    plt.show()

    xexact_ = np.interp(t, texact, xexact)
    vexact_ = np.interp(t, texact, vexact)

    data = {
        't': t,
        'xmap': x,
        'xmean': x,
        'xexact': xexact_,
        'x05': xlo,
        'x95': xhi,
        'vmap': v,
        'vmean': v,
        'vexact': vexact_,
        'v05': vlo,
        'v95': vhi
    }

    df = pd.DataFrame(data)
    df.to_csv('uq_pred.csv', index=False)

    data = {
        't': td,
        'x': xd
    }

    df = pd.DataFrame(data)
    df.to_csv('uq_data.csv', index=False)

    data = {
        'omegasq_mean': [np.mean(samples[:,2]**2)],
        'omegasq_std': [np.std(samples[:,2]**2)]
    }

    df = pd.DataFrame(data)
    df.to_csv('uq_omega.csv', index=False)


if __name__ == '__main__':
    main()
