#!/usr/bin/env python3

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import torch
from torch.optim import Adam
from tqdm import tqdm

from bodil.HMC import HMC
from back_flows import CubeNoLid

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_traj', type=str, help="path of beads trajectories, in csv format.")
    parser.add_argument('--vfield-path', type=str, default='vfield', help="path Stokes solutions, in npy format.")
    args = parser.parse_args()

    L = 5 # cm
    omega_max = 0.45 # rad/s

    seed = 2349873
    beta = 840000
    sigma_data = np.array([0.05, 0.05, 0.07])
    lr = 1e-4
    num_epochs = 5001
    stats_every = 500
    num_samples = 10000
    rng = np.random.default_rng(seed=seed)

    csv_path = args.csv_traj

    df = pd.read_csv(csv_path)

    t = df['t'].to_numpy() # s
    nbeads = len([name for name in df.columns if 'bead' in name]) // 3
    nmotors = len([name for name in df.columns if 'omega' in name])
    nt = len(t)

    data_beads = np.empty((nbeads, nt, 3))
    omegas = np.empty((nmotors, nt))

    for j in range(nbeads):
        for dim, code in enumerate(['x', 'y', 'z']):
            data_beads[j,:,dim] = df[f"bead{j}{code}"].to_numpy()

    for k in range(nmotors):
        omegas[k,:] = df[f"omega{k}"].to_numpy()

    omegas *= omega_max

    vfield_path = args.vfield_path

    flow = CubeNoLid(path_x0=os.path.join(vfield_path, 'nolid_x0.npy'),
                     path_z0=os.path.join(vfield_path, 'nolid_z0.npy'))


    t_ = torch.from_numpy(t).float()
    omegas_ = torch.from_numpy(omegas).float()
    data_beads_ = torch.from_numpy(data_beads).float()


    # flattened solution: beads trajectories; z-sed velocities
    u_ = torch.zeros((nbeads * nt * 3) + nbeads)

    # initial guess
    u_[:nbeads*nt*3] = data_beads_.flatten()
    u_[nbeads*nt*3:nbeads*nt*3+nbeads] = -0.05 # sedz

    u_.requires_grad = True

    def neg_log_posterior(u_):
        beads_ = u_[:nbeads*nt*3].reshape((nbeads, nt, 3))
        sedz = u_[nbeads*nt*3:nbeads*nt*3+nbeads]

        nlp = 0

        for j in range(nbeads):
            x = beads_[j,:,0]
            y = beads_[j,:,1]
            z = beads_[j,:,2]

            # pde
            vx, vy, vz = flow.get_velocity(x, y, z, omegas_)

            dt = torch.diff(t_)
            dxdt = (x[1:] - x[:-1]) / dt
            dydt = (y[1:] - y[:-1]) / dt
            dzdt = (z[1:] - z[:-1]) / dt

            sedvz = torch.zeros(nt-1)
            idx = torch.argwhere(z[:-1] > 0.09)
            sedvz[idx] = sedz[j]

            resx = dxdt - vx[:-1]
            resy = dydt - vy[:-1]
            resz = dzdt - vz[:-1] - sedvz

            nlp += beta * torch.mean(resx**2 + resy**2 + resz**2)

            # data
            dresx = x - data_beads_[j,:,0]
            dresy = y - data_beads_[j,:,1]
            dresz = z - data_beads_[j,:,2]

            nlp += torch.sum(dresx**2 / (2 * sigma_data[0]**2)) + nt/2 * np.log(2 * np.pi * sigma_data[0]**2)
            nlp += torch.sum(dresy**2 / (2 * sigma_data[1]**2)) + nt/2 * np.log(2 * np.pi * sigma_data[1]**2)
            nlp += torch.sum(dresz**2 / (2 * sigma_data[2]**2)) + nt/2 * np.log(2 * np.pi * sigma_data[2]**2)

        return nlp


    optim = Adam([u_], lr=lr)

    for epoch in range(num_epochs):
        optim.zero_grad()
        loss = neg_log_posterior(u_)
        loss.backward()
        optim.step()
        if epoch % stats_every == 0:
            sedz = u_[nbeads*nt*3:nbeads*nt*3+nbeads].detach().numpy()
            print(f"epoch {epoch:06d}: loss {loss.item():.4e}, sedz={sedz}")


    hmc = HMC([u_], dt=0.00028, L=10, M=0.1)

    def closure():
        hmc.zero_grad()
        U = neg_log_posterior(u_)
        U.backward()
        return U

    samples = []
    num_accepted = 0
    print("HMC sampling...")
    for k in tqdm(range(num_samples)):
        y_, H_, U_, accepted = hmc.step(closure)
        samples.append(y_[0].detach().numpy())
        num_accepted += accepted

    print(f"accptance rate: {num_accepted/num_samples}")
    samples = np.array(samples)

    # estimate covariance
    mean = np.mean(samples, axis=0)
    X = samples - mean[None, :]
    cov = (X.T @ X) / (num_samples - 1)

    with open('hmc_cov.npy', 'wb') as f:
        np.save(f, cov)

    u_mean = np.mean(samples, axis=0)
    u_lo = np.quantile(samples, q=0.05, axis=0)
    u_hi = np.quantile(samples, q=0.95, axis=0)

    beads_mean = u_mean[:nbeads*nt*3].reshape((nbeads, nt, 3)) * L
    beads_lo = u_lo[:nbeads*nt*3].reshape((nbeads, nt, 3)) * L
    beads_hi = u_hi[:nbeads*nt*3].reshape((nbeads, nt, 3)) * L

    data = {
        't': t
    }

    for j in range(nbeads):
        for dim, code in enumerate(['x', 'y', 'z']):
            data[f"bead{j}{code}_mean"] = beads_mean[j,:,dim]
            data[f"bead{j}{code}_q05"] = beads_lo[j,:,dim]
            data[f"bead{j}{code}_q95"] = beads_hi[j,:,dim]
            data[f"bead{j}{code}_data"] = data_beads[j,:,dim] * L

    pd.DataFrame(data).to_csv("hmc_beads.csv", index=False)

    fig, axes = plt.subplots(ncols=nbeads, figsize=(nbeads * 4.8,  3.6), sharey=True)

    for j in range(nbeads):
        ax = axes[j]
        for dim, code in enumerate(['x', 'y', 'z']):
            ax.fill_between(t, beads_lo[j,:,dim], beads_hi[j,:,dim],
                            lw=0, alpha=0.2, color=f'C{dim}',
                            label='5-95% quantiles of posterior')
            ax.plot(t, beads_mean[j,:,dim],
                    color=f'C{dim}', ls='-',
                    label='mean')
            ax.plot(t, data_beads[j,:,dim] * L,
                    color=f'C{dim}', ls='--',
                    label='data')

        ax.set_xlabel(r'$t$ (s)')
        ax.set_ylim(0, L)
        if j == 0:
            #ax.legend()
            ax.set_ylabel('coordinates (cm)')

    plt.tight_layout()
    plt.show()
    plt.close()

    vz_samples = samples[:,nbeads*nt*3:nbeads*nt*3+nbeads] * L

    data = {}
    for j in range(nbeads):
        data[f'vz{j}'] = vz_samples[:,j]
    pd.DataFrame(data).to_csv("hmc_vz.csv", index=False)

    fig, ax = plt.subplots()
    for j in range(nbeads):
        ax.hist(vz_samples[:,j], label=f'bead {j}', density=True, bins=25)
    ax.set_ylim(0, None)
    ax.set_xlabel(r'$v_\mathrm{sed}$ (cm/s)')
    ax.set_ylabel(r'$p(v_\mathrm{sed})$')
    ax.legend()

    plt.tight_layout()
    plt.show()
    plt.close()


if __name__ == '__main__':
    main()
