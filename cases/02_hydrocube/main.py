#!/usr/bin/env python3

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import torch
from torch.optim import Adam

from back_flows import CubeNoLid

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_traj', type=str, help="path of beads trajectories, in csv format.")
    parser.add_argument('--vfield-path', type=str, default='vfield', help="path Stokes solutions, in npy format.")
    args = parser.parse_args()

    L = 5 # cm
    omega_max = 0.45 # rad/s

    seed = 2349873
    sigma_pde = 0.01
    sigma_data = np.array([0.05, 0.05, 0.07])
    lr = 1e-4
    num_epochs = 5000
    stats_every = 500
    num_samples = 5000
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

            nlp += torch.sum(resx**2 / (2 * sigma_pde**2)) + (nt-1)/2 * np.log(2 * np.pi * sigma_pde**2)
            nlp += torch.sum(resy**2 / (2 * sigma_pde**2)) + (nt-1)/2 * np.log(2 * np.pi * sigma_pde**2)
            nlp += torch.sum(resz**2 / (2 * sigma_pde**2)) + (nt-1)/2 * np.log(2 * np.pi * sigma_pde**2)

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


    # Laplace approximation
    H = torch.autograd.functional.hessian(neg_log_posterior, u_, create_graph=True)
    H = H.detach().numpy()

    uMAP = u_.detach().numpy()
    cov = np.linalg.inv(H)

    var = np.diag(cov)

    q_5_95_std = 2.5758

    beads_var = var[:nbeads*nt*3].reshape((nbeads, nt, 3)) * L**2
    obs_std = np.sqrt(beads_var + sigma_data[None,None,:]**2)

    beads_MAP = uMAP[:nbeads*nt*3].reshape((nbeads, nt, 3)) * L
    beads_lo = beads_MAP - q_5_95_std * obs_std
    beads_hi = beads_MAP + q_5_95_std * obs_std

    fig, axes = plt.subplots(ncols=nbeads, figsize=(nbeads * 4.8,  3.6), sharey=True)

    for j in range(nbeads):
        ax = axes[j]
        for dim, code in enumerate(['x', 'y', 'z']):
            ax.fill_between(t, beads_lo[j,:,dim], beads_hi[j,:,dim],
                            lw=0, alpha=0.2, color=f'C{dim}',
                            label='5-95% quantiles of posterior')
            ax.plot(t, beads_MAP[j,:,dim],
                    color=f'C{dim}', ls='-',
                    label='MAP')
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


    sed_MAP = uMAP[nbeads*nt*3:nbeads*nt*3+nbeads] * L
    sed_std = np.sqrt(var[nbeads*nt*3:nbeads*nt*3+nbeads]) * L

    lo = np.min(sed_MAP - 3 * sed_std)
    hi = np.max(sed_MAP + 3 * sed_std)

    fig, ax = plt.subplots()
    for j in range(nbeads):
        mu = sed_MAP[j]
        sigma = sed_std[j]
        vz = np.linspace(lo, hi, 1000, endpoint=True)
        pz = np.exp(-(vz-mu)**2/(2*sigma**2)) / np.sqrt(2 * np.pi * sigma**2)
        ax.plot(vz, pz, label=f'bead {j}')
    ax.set_xlim(lo, hi)
    ax.set_ylim(0, None)
    ax.set_xlabel(r'$v_\mathrm{sed}$ (cm/s)')
    ax.set_ylabel(r'$p(v_\mathrm{sed})$')
    ax.legend()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
