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

    seed = 2349873
    sigma_pde = 0.005
    lr = 1e-4
    num_epochs = 15000
    num_samples = 5000
    rng = np.random.default_rng(seed=seed)

    csv_path = args.csv_traj

    df = pd.read_csv(csv_path)

    t = df['t'].to_numpy()
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


    vfield_path = args.vfield_path

    flow = CubeNoLid(path_x0=os.path.join(vfield_path, 'nolid_x0.npy'),
                     path_z0=os.path.join(vfield_path, 'nolid_z0.npy'))


    t_ = torch.from_numpy(t).float()
    omegas_ = torch.from_numpy(omegas).float()
    data_beads_ = torch.from_numpy(data_beads).float()


    # flattened solution: beads trajectories; z-sed velocities; sigmaxyz
    u_ = torch.zeros((nbeads * nt * 3) + nbeads + 3)

    # initial guess
    u_[:nbeads*nt*3] = data_beads_.flatten()
    u_[nbeads*nt*3:nbeads*nt*3+nbeads] = -0.05 # sedz
    u_[nbeads*nt*3+nbeads+0] = 0.01 # sigma x
    u_[nbeads*nt*3+nbeads+1] = 0.01 # sigma y
    u_[nbeads*nt*3+nbeads+2] = 0.05 # sigma z

    u_.requires_grad = True

    def neg_log_posterior(u_):
        beads_ = u_[:nbeads*nt*3].reshape((nbeads, nt, 3))
        sedz = u_[nbeads*nt*3:nbeads*nt*3+nbeads]
        sigma_data = u_[nbeads*nt*3+nbeads:]

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

            resx = dxdt - vx[:-1]
            resy = dydt - vy[:-1]
            resz = dzdt - vz[:-1] - sedz[j]

            nlp += torch.sum(resx**2 / (2 * sigma_pde**2)) + (nt-1)/2 * np.log(2 * np.pi * sigma_pde**2)
            nlp += torch.sum(resy**2 / (2 * sigma_pde**2)) + (nt-1)/2 * np.log(2 * np.pi * sigma_pde**2)
            nlp += torch.sum(resz**2 / (2 * sigma_pde**2)) + (nt-1)/2 * np.log(2 * np.pi * sigma_pde**2)

            # data
            dresx = x - data_beads_[j,:,0]
            dresy = y - data_beads_[j,:,1]
            dresz = z - data_beads_[j,:,2]

            nlp += torch.sum(dresx**2 / (2 * sigma_data[0]**2)) + nt/2 * torch.log(2 * np.pi * sigma_data[0]**2)
            nlp += torch.sum(dresy**2 / (2 * sigma_data[1]**2)) + nt/2 * torch.log(2 * np.pi * sigma_data[1]**2)
            nlp += torch.sum(dresz**2 / (2 * sigma_data[2]**2)) + nt/2 * torch.log(2 * np.pi * sigma_data[2]**2)

        return nlp


    optim = Adam([u_], lr=lr)

    for epoch in range(num_epochs):
        optim.zero_grad()
        loss = neg_log_posterior(u_)
        loss.backward()
        optim.step()
        if epoch % 100 == 0:
            sedz = u_[nbeads*nt*3:nbeads*nt*3+nbeads].detach().numpy()
            sigs = u_[nbeads*nt*3+nbeads:].detach().numpy()

            print(f"epoch {epoch:06d}: loss {loss.item():.4e}, sigma={sigs}, sedz={sedz}")


    # Laplace approximation
    H = torch.autograd.functional.hessian(neg_log_posterior, u_, create_graph=True)
    H = H.detach().numpy()

    uMAP = u_.detach().numpy()

    samples = np.zeros((len(uMAP), num_samples))

    eigvals, eigvecs = np.linalg.eig(H)
    print(np.sort(eigvals)[:50])

    for k in range(num_samples):
        z = rng.normal(0, 1/np.sqrt(np.maximum(eigvals, 1e-8)), len(uMAP))
        samples[:,k] = uMAP + eigvecs @ z

    umean = np.mean(samples, axis=1)
    ulo = np.quantile(samples, q=0.05, axis=1)
    uhi = np.quantile(samples, q=0.95, axis=1)

    beads_MAP = uMAP[:nbeads*nt*3].reshape((nbeads, nt, 3))
    beads_lo = ulo[:nbeads*nt*3].reshape((nbeads, nt, 3))
    beads_hi = uhi[:nbeads*nt*3].reshape((nbeads, nt, 3))

    fig, axes = plt.subplots(ncols=nbeads, figsize=(nbeads * 4.8,  3.6))

    for j in range(nbeads):
        ax = axes[j]
        for dim, code in enumerate(['x', 'y', 'z']):
            ax.fill_between(t, beads_lo[j,:,dim], beads_hi[j,:,dim],
                            lw=0, alpha=0.2, color=f'C{dim}',
                            label='5-95% quantiles of posterior')
            ax.plot(t, beads_MAP[j,:,dim], color=f'C{dim}', ls='-')
            ax.plot(t, data_beads[j,:,dim], color=f'C{dim}', ls='none', marker='.')

        ax.set_xlabel(r'$t$')
        ax.set_ylim(0, 1)
        ax.set_ylabel('position')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
