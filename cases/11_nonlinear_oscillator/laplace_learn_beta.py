#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.optim import Adam
from scipy.stats import norm

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
    t = 0.0
    xs = [x0]
    vs = [v0]
    ts = [t]

    x, v = x0, v0
    for _ in range(nt):
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
    # data-generating system (Duffing)
    T = 20.0
    k1 = 1.0
    k2 = 10.0
    m = 15.0
    x0 = 2.0
    v0 = 0.0

    seed = 2349873
    num_epochs = 100000
    lr_y = 5e-4
    lr_logsig = 5e-4
    num_data = 20
    sigma_data = 0.1
    rng = np.random.default_rng(seed=seed)

    td, xd_np, texact, xexact, vexact = generate_data(
        nt=10000, num_data=num_data,
        T=T, m=m, k1=k1, k2=k2,
        x0=x0, v0=v0,
        rng=rng, sigma=sigma_data
    )

    # inference grid
    nt = 255
    t = np.linspace(0, T, nt + 1, endpoint=True)
    dt = t[1] - t[0]

    # torch data
    td_ids = torch.from_numpy((td / dt).astype(int))
    xd = torch.from_numpy(xd_np)

    yprev = None

    # --- Single run: learn sigma_ode (thus beta) instead of looping over beta ---
    print("Learning sigma_ode (physics noise) jointly with y ...")

    if yprev is None:
        y = torch.zeros(1 + (nt + 1) * 2, requires_grad=True)
    else:
        y = torch.from_numpy(yprev).requires_grad_()

    # learn log(sigma_ode)
    log_sigma_ode = torch.tensor(np.log(1e-4), dtype=y.dtype, requires_grad=True)

    # optimizer (optionally separate lrs)
    optim = Adam(
        [{'params': [y], 'lr': lr_y},
         {'params': [log_sigma_ode], 'lr': lr_logsig}]
    )

    def neg_log_posterior(y, log_sigma_ode):
        """
        Physics residual treated as Gaussian noise with unknown sigma_ode.
        Uses *sum* form + normalization term to make sigma_ode (thus beta) identifiable.
        """
        sigma_ode = torch.exp(log_sigma_ode)

        omegasq = y[0]
        x = y[1:nt+2]
        v = y[nt+2:]

        dxdt = torch.diff(x) / dt
        dvdt = torch.diff(v) / dt
        xm = (x[:-1] + x[1:]) / 2
        vm = (v[:-1] + v[1:]) / 2

        r1 = dxdt - vm
        r2 = dvdt + omegasq * xm
        r = torch.cat([r1, r2], dim=0)
        Nr = r.numel()

        # physics NLL (Gaussian) including normalization
        nlp_phys = 0.5 * torch.mean((r / sigma_ode) ** 2) + log_sigma_ode

        # data NLL (Gaussian) including constants
        data_res = x[td_ids] - xd
        n_data = data_res.numel()
        nlp_data = 0.5 * torch.sum((data_res / sigma_data) ** 2) \
                   + n_data * np.log(sigma_data) + 0.5 * n_data * np.log(2 * np.pi)

        return nlp_phys + nlp_data

    losses = []
    for epoch in range(num_epochs):
        optim.zero_grad()
        loss = neg_log_posterior(y, log_sigma_ode)
        loss.backward()
        optim.step()

        l = float(loss.detach().cpu().float())
        losses.append(l)

        if epoch % 10000 == 0:
            sigma_ode_val = float(torch.exp(log_sigma_ode).detach().cpu())
            beta_val = 1.0 / (sigma_ode_val**2)
            print(f"epoch {epoch:06d}, loss {l:.6e}, sigma_ode {sigma_ode_val:.3e}, beta {beta_val:.3e}")

    # detach results
    y_np = y.detach().cpu().numpy()
    omegasq = y_np[0]
    x = y_np[1:nt+2]
    v = y_np[nt+2:]

    sigma_ode_learned = float(torch.exp(log_sigma_ode).detach().cpu().numpy())
    beta_learned = 1.0 / (sigma_ode_learned ** 2)

    print(f"Learned sigma_ode = {sigma_ode_learned:.6e}  (beta = {beta_learned:.6e})")

    # Hessian w.r.t y only (Laplace around MAP for y, holding log_sigma_ode fixed)
    H_y = torch.autograd.functional.hessian(
        lambda yy: neg_log_posterior(yy, log_sigma_ode.detach()),
        y,
        create_graph=False
    )
    H = H_y.detach().cpu().numpy()
    H = 0.5 * (H + H.T)  # symmetrize

    # invert with jitter for stability
    eps = 1e-8 * np.max(np.abs(np.diag(H))) + 1e-12
    try:
        cov = np.linalg.inv(H + eps * np.eye(H.shape[0]))
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(H + eps * np.eye(H.shape[0]))

    # marginals from Laplace covariance
    std = np.sqrt(np.maximum(np.diag(cov), 0.0))

    z_lo = norm.ppf(0.05)
    z_hi = norm.ppf(0.95)

    omegasq_mean = omegasq
    omegasq_lo = omegasq + z_lo * std[0]
    omegasq_hi = omegasq + z_hi * std[0]

    xmean = x
    xlo = x + z_lo * std[1:1+len(x)]
    xhi = x + z_hi * std[1:1+len(x)]

    vmean = v
    vlo = v + z_lo * std[1+len(x):]
    vhi = v + z_hi * std[1+len(x):]

    # plot
    fig, axes = plt.subplots(ncols=2, figsize=(9.6, 4.8))

    ax = axes[0]
    ax.fill_between(t, xlo, xhi, lw=0, alpha=0.2, color='r',
                    label='5-95% quantiles of posterior (Laplace)')
    ax.plot(t, x, '-r', label='MAP')
    ax.plot(texact, xexact, '--k', label='exact')
    ax.plot(td, xd.detach().numpy(), '+k', label='data')
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$x$")
    ax.set_xlim(0, T)
    ax.set_ylim(-3, 3)
    ax.legend(frameon=False)

    ax = axes[1]
    ax.fill_between(t, vlo, vhi, lw=0, alpha=0.2, color='r')
    ax.plot(t, v, '-r', label='MAP')
    ax.plot(texact, vexact, '--k', label='exact')
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$v$")
    ax.set_xlim(0, T)
    ax.set_ylim(-3, 3)

    plt.tight_layout()
    plt.show()

    # save CSVs (similar to your previous naming, but keyed by learned sigma/beta)
    xexact_ = np.interp(t, texact, xexact)
    vexact_ = np.interp(t, texact, vexact)

    pred = {
        't': t,
        'xmap': x,
        'xmean': xmean,
        'xexact': xexact_,
        'x05': xlo,
        'x95': xhi,
        'vmap': v,
        'vmean': vmean,
        'vexact': vexact_,
        'v05': vlo,
        'v95': vhi
    }
    pd.DataFrame(pred).to_csv(f'laplace_pred_beta_learned_{beta_learned:.6e}.csv', index=False)

    meas = {'t': td, 'x': xd.detach().numpy()}
    pd.DataFrame(meas).to_csv(f'laplace_data_beta_learned_{beta_learned:.6e}.csv', index=False)

    omega = {
        'omegasq_mean': [float(omegasq_mean)],
        'omegasq_lo05': [float(omegasq_lo)],
        'omegasq_hi95': [float(omegasq_hi)],
        'omegasq_std': [float(np.sqrt(cov[0, 0]))],
        'sigma_ode': [sigma_ode_learned],
        'beta_implied': [beta_learned]
    }
    pd.DataFrame(omega).to_csv(f'laplace_omega_beta_learned_{beta_learned:.6e}.csv', index=False)


if __name__ == '__main__':
    main()
