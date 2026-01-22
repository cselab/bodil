#!/usr/bin/env python3

import argparse
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


def parse_args():
    p = argparse.ArgumentParser(description="B-ODIL with beta selection using validation data")
    p.add_argument("--num-data", type=int, default=50, help="Number of measurements")
    p.add_argument(
        "--datagen",
        choices=["duffing", "linear"],
        default="duffing",
        help="Data generation model (duffing => k2>0, linear => k2=0)",
    )
    p.add_argument("--no-plot", action="store_true", help="Disable plotting")
    return p.parse_args()


def main():
    args = parse_args()

    T = 20.0
    if args.datagen == "duffing":
        datagen = "Duffing"
        k1 = 1.0
        k2 = 10.0
        m = 15.0
        x0 = 2.0
        v0 = 0.0
        sigma_data = 0.4
    else:
        datagen = "linear"
        k1 = 1.0
        k2 = 0.0
        m = 1.0
        x0 = 0.5
        v0 = 0.2
        sigma_data = 0.1

    seed = 2349873

    num_samples = 100000
    burnin = 100000
    step0 = 0.01                   # initial proposal std
    target_acc = 0.35              # good for d=3
    adapt_interval = 2000          # how often to update step during burn-in
    gamma0 = 0.5                   # adaptation gain (will decay)

    num_data = args.num_data
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

    # --- Adaptive burn-in (Robbins–Monro), then fixed-step sampling ---
    step = step0
    log_step = np.log(step)

    samples = []
    accepted_post = 0

    # tracking acceptance during burn-in for adaptation
    acc_window = 0
    win_count = 0

    for i in range(num_samples + burnin):
        # propose
        thetap = rng.normal(theta, step)
        logpp = log_posterior(thetap)

        loga = logpp - logp
        if np.log(rng.uniform()) < loga:
            theta = thetap
            logp = logpp
            accepted = True
        else:
            accepted = False

        # --- burn-in adaptation only ---
        if i < burnin:
            acc_window += int(accepted)
            win_count += 1

            if (i + 1) % adapt_interval == 0:
                acc_rate = acc_window / win_count

                # diminishing adaptation gain
                # (keeps it stable and makes it easier to argue correctness)
                gamma = gamma0 / np.sqrt((i + 1) / adapt_interval)

                # Robbins–Monro update on log(step)
                log_step += gamma * (acc_rate - target_acc)
                # clamp to avoid crazy steps
                log_step = np.clip(log_step, np.log(1e-6), np.log(1.0))
                step = float(np.exp(log_step))

                if (i + 1) % (10000) == 0:
                    print(f"burnin step {i+1:6d}: acc={acc_rate:.3f}, step={step:.3e}")

                # reset window
                acc_window = 0
                win_count = 0

        else:
            # --- sampling phase: fixed step size ---
            accepted_post += int(accepted)
            samples.append(theta.copy())

    print(f"final proposal step (after burn-in): {step:.3e}")
    print(f"post-burn-in acceptance rate: {accepted_post / num_samples:.3f}")

    samples = np.array(samples)

    logps = np.array([log_posterior(s) for s in samples])
    theta_map = samples[np.argmax(logps)]
    _, x_map, v_map = forward(theta_map[0], theta_map[1], theta_map[2])

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

    if not args.no_plot:
        fig, axes = plt.subplots(ncols=2, figsize=(9.6,4.8))
        ax = axes[0]
        ax.fill_between(t, xlo, xhi, lw=0, alpha=0.2, color='r', label='5-95% quantiles of posterior')
        ax.plot(t, x_map, '-r', label='MAP')
        ax.plot(texact, xexact, '--k', label='exact')
        ax.plot(td, xd, '+k', label='data')
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(r"$x$")
        ax.set_xlim(0, T)
        #ax.set_ylim(-1.5, 1.5)
        ax.legend(frameon=False)

        ax = axes[1]
        ax.fill_between(t, vlo, vhi, lw=0, alpha=0.2, color='r')
        ax.plot(t, v_map, '-r')
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
        'xmap': x_map,
        'xmean': xmean,
        'xexact': xexact_,
        'x05': xlo,
        'x95': xhi,
        'vmap': v_map,
        'vmean': vmean,
        'vexact': vexact_,
        'v05': vlo,
        'v95': vhi
    }

    df = pd.DataFrame(data)
    df.to_csv(f'uq_pred_{datagen}_N_{num_data}.csv', index=False)

    data = {
        't': td,
        'x': xd
    }

    df = pd.DataFrame(data)
    df.to_csv(f'uq_data_{datagen}_N_{num_data}.csv', index=False)


if __name__ == '__main__':
    main()
