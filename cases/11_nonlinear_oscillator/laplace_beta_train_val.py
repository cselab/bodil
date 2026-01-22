#!/usr/bin/env python3
"""
B-ODIL with beta selection using validation data.
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.optim import Adam
from scipy.stats import norm

def generate_data(nt, num_data, T, m, k1, k2, x0, v0, rng, sigma):
    dt = T / nt
    t = 0.0
    xs = [x0]
    vs = [v0]
    ts = [t]

    x, v = x0, v0
    for _ in range(nt):
        # Leapfrog
        F = -(k1 * x + k2 * x**3)
        v += dt / 2 * F / m
        x += dt * v
        F = -(k1 * x + k2 * x**3)
        v += dt / 2 * F / m
        t += dt
        ts.append(t)
        xs.append(x)
        vs.append(v)

    xth = np.array(xs)
    vth = np.array(vs)
    tth = np.array(ts)

    # measurements
    t_meas = rng.uniform(0.0, T / 2, num_data)
    x_true = np.interp(t_meas, tth, xth)
    x_meas = x_true + sigma * rng.standard_normal(num_data)

    return t_meas, x_meas, tth, xth, vth


def nlpredictive_gaussian(x_obs, mu, var):
    """Sum of negative log predictive density for N(mu, var)."""
    var = np.maximum(var, 1e-12)
    return float(np.sum(0.5 * (x_obs - mu) ** 2 / var + 0.5 * np.log(2.0 * np.pi * var)))


def coverage_90(x_obs, mu, var):
    z_lo = norm.ppf(0.05)
    z_hi = norm.ppf(0.95)
    s = np.sqrt(np.maximum(var, 1e-12))
    lo = mu + z_lo * s
    hi = mu + z_hi * s
    return float(np.mean((x_obs >= lo) & (x_obs <= hi)))


def fit_map_and_laplace(
    beta,
    nt,
    t,
    dt,
    td_ids_train_t,
    xd_train_t,
    sigma_data,
    num_epochs,
    lr,
    y_init_np=None,
    dtype=torch.float64,
):
    """
    Fit MAP y using TRAIN data only, then compute Laplace covariance via Hessian at MAP.
    Returns:
      y_map_np, cov_np, losses
    """
    D = 1 + 2 * (nt + 1)
    if y_init_np is None:
        y = torch.zeros(D, dtype=dtype, requires_grad=True)
    else:
        y = torch.from_numpy(y_init_np).to(dtype=dtype).requires_grad_()

    optim = Adam([y], lr=lr)

    def neg_log_posterior(yvec):
        omegasq = yvec[0]
        x = yvec[1 : nt + 2]
        v = yvec[nt + 2 :]

        dxdt = torch.diff(x) / dt
        dvdt = torch.diff(v) / dt
        xm = (x[:-1] + x[1:]) / 2
        vm = (v[:-1] + v[1:]) / 2

        ode1_res = dxdt - vm
        ode2_res = dvdt + omegasq * xm

        # physics penalty (sum form for consistent scaling w.r.t. nt)
        phys = 0.5 * beta * torch.sum(ode1_res**2 + ode2_res**2)

        # TRAIN data likelihood (Gaussian, includes constants)
        data_res = x[td_ids_train_t] - xd_train_t
        n_data = data_res.numel()
        data = 0.5 * torch.sum((data_res / sigma_data) ** 2) \
               + n_data * np.log(sigma_data) + 0.5 * n_data * np.log(2.0 * np.pi)

        return phys + data

    losses = []
    for epoch in range(num_epochs+1):
        optim.zero_grad()
        loss = neg_log_posterior(y)
        loss.backward()
        optim.step()
        l = float(loss.detach().cpu().numpy())
        losses.append(l)
        if epoch % 10000 == 0:
            print(f"  beta {beta:.1e} epoch {epoch:06d} loss {l:.6e}")

    # Hessian at MAP (w.r.t y only)
    H = torch.autograd.functional.hessian(neg_log_posterior, y, create_graph=False)
    H = H.detach().cpu().numpy()
    H = 0.5 * (H + H.T)

    # stable inversion with jitter
    diag_scale = np.max(np.abs(np.diag(H))) if H.shape[0] else 1.0
    eps = 1e-8 * diag_scale + 1e-12
    try:
        cov = np.linalg.inv(H + eps * np.eye(H.shape[0]))
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(H + eps * np.eye(H.shape[0]))

    y_map = y.detach().cpu().numpy()
    return y_map, cov, losses


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
    else:
        datagen = "linear"
        k1 = 1.0
        k2 = 0.0
        m = 1.0
        x0 = 0.5
        v0 = 0.2

    seed = 2349873
    rng = np.random.default_rng(seed=seed)

    # optimization
    num_epochs = 50000
    lr = 5e-4

    # measurements
    num_data = args.num_data
    sigma_data = 0.1

    td, xd_np, texact, xexact, vexact = generate_data(
        nt=10000,
        num_data=num_data,
        T=T,
        m=m,
        k1=k1,
        k2=k2,
        x0=x0,
        v0=v0,
        rng=rng,
        sigma=sigma_data,
    )

    nt = 255
    t = np.linspace(0, T, nt + 1, endpoint=True)
    dt = t[1] - t[0]

    # split measurements into TRAIN and VAL
    perm = rng.permutation(num_data)
    n_val = max(1, num_data // 5)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    td_train = td[train_idx]
    xd_train = xd_np[train_idx]
    td_val = td[val_idx]
    xd_val = xd_np[val_idx]

    td_ids_train = (td_train / dt).astype(int)
    td_ids_val = (td_val / dt).astype(int)

    dtype = torch.float64
    td_ids_train_t = torch.from_numpy(td_ids_train)
    xd_train_t = torch.from_numpy(xd_train).to(dtype=dtype)

    # Beta grid search via VAL predictive NLPD
    beta_grid = np.logspace(-2, 7, 50) # from 1e-2 to 1e7
    results = []

    yprev = None
    best = None

    print(f"TRAIN size: {len(train_idx)}, VAL size: {len(val_idx)}")
    for beta in beta_grid:
        print(f"\nFitting beta = {beta:.1e} (TRAIN only)")
        y_map, cov, _losses = fit_map_and_laplace(
            beta=beta,
            nt=nt,
            t=t,
            dt=dt,
            td_ids_train_t=td_ids_train_t,
            xd_train_t=xd_train_t,
            sigma_data=sigma_data,
            num_epochs=num_epochs,
            lr=lr,
            y_init_np=yprev,
            dtype=dtype,
        )

        # unpack MAP
        omegasq = y_map[0]
        x_map = y_map[1 : nt + 2]
        v_map = y_map[nt + 2 :]

        # predictive variance for x at each grid point (Laplace)
        x_var = np.diag(cov)[1 : 1 + (nt + 1)]
        # predictive for observed measurements includes measurement noise
        mu_val = x_map[td_ids_val]
        var_val = x_var[td_ids_val] + sigma_data**2

        nlpd = nlpredictive_gaussian(xd_val, mu_val, var_val)
        cov90 = coverage_90(xd_val, mu_val, var_val)

        print(f"  beta {beta:.1e}  VAL NLPD: {nlpd:.6f}   VAL 90% coverage: {cov90:.3f}")
        results.append((beta, nlpd, cov90))

        if best is None or nlpd < best[1]:
            best = (beta, nlpd, cov90, y_map.copy())
        yprev = y_map.copy()  # warm start next beta

    best_beta, best_nlpd, best_cov90, best_yinit = best
    print("\n==============================")
    print(f"Selected beta = {best_beta:.1e} (lowest VAL NLPD = {best_nlpd:.6f}, coverage90={best_cov90:.3f})")
    print("==============================\n")

    # Save beta-sweep summary
    df_beta = pd.DataFrame(
        {"beta": [b for b, _, _ in results], "val_nlpd": [n for _, n, _ in results], "val_cov90": [c for _, _, c in results]}
    )

    df_beta.to_csv(f"beta_selection_summary_{datagen}_N_{num_data}.csv", index=False)

    # Final fit on ALL data with chosen beta
    td_ids_all = (td / dt).astype(int)
    td_ids_all_t = torch.from_numpy(td_ids_all)
    xd_all_t = torch.from_numpy(xd_np).to(dtype=dtype)

    print(f"Refitting with selected beta = {best_beta:.1e} on ALL data")
    y_final, cov_final, _ = fit_map_and_laplace(
        beta=best_beta,
        nt=nt,
        t=t,
        dt=dt,
        td_ids_train_t=td_ids_all_t,  # reuse function; "train" now means "all"
        xd_train_t=xd_all_t,
        sigma_data=sigma_data,
        num_epochs=num_epochs,
        lr=lr,
        y_init_np=best_yinit,
        dtype=dtype,
    )

    omegasq = y_final[0]
    x = y_final[1 : nt + 2]
    v = y_final[nt + 2 :]

    # Laplace marginals (5%/95%) for state variables
    std = np.sqrt(np.maximum(np.diag(cov_final), 0.0))
    z_lo = norm.ppf(0.05)
    z_hi = norm.ppf(0.95)

    omegasq_mean = omegasq
    omegasq_lo = omegasq + z_lo * std[0]
    omegasq_hi = omegasq + z_hi * std[0]

    xlo = x + z_lo * std[1 : 1 + len(x)]
    xhi = x + z_hi * std[1 : 1 + len(x)]

    vlo = v + z_lo * std[1 + len(x) :]
    vhi = v + z_hi * std[1 + len(x) :]

    if not args.no_plot:
        fig, axes = plt.subplots(ncols=2, figsize=(9.6, 4.8))
        ax = axes[0]
        ax.fill_between(t, xlo, xhi, lw=0, alpha=0.2, color="r", label="5–95% (Laplace)")
        ax.plot(t, x, "-r", label="MAP")
        ax.plot(texact, xexact, "--k", label="exact")
        ax.plot(td, xd_np, "+k", label="data")
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(r"$x$")
        ax.set_xlim(0, T)
        ax.set_ylim(-3, 3)
        ax.legend(frameon=False)
        ax.set_title(f"Selected beta = {best_beta:.1e}")

        ax = axes[1]
        ax.fill_between(t, vlo, vhi, lw=0, alpha=0.2, color="r")
        ax.plot(t, v, "-r")
        ax.plot(texact, vexact, "--k")
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(r"$v$")
        ax.set_xlim(0, T)
        ax.set_ylim(-3, 3)

        plt.tight_layout()
        plt.show()

    xexact_ = np.interp(t, texact, xexact)
    vexact_ = np.interp(t, texact, vexact)

    pred = {
        "t": t,
        "xmap": x,
        "xmean": x,
        "xexact": xexact_,
        "x05": xlo,
        "x95": xhi,
        "vmap": v,
        "vmean": v,
        "vexact": vexact_,
        "v05": vlo,
        "v95": vhi,
    }
    pd.DataFrame(pred).to_csv(f"laplace_pred_{datagen}_N_{num_data}.csv", index=False)

    meas = {"t": td, "x": xd_np}
    pd.DataFrame(meas).to_csv(f"laplace_data_{datagen}_N_{num_data}.csv", index=False)


if __name__ == "__main__":
    main()
