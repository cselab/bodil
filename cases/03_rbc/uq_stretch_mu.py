#!/usr/bin/env python3

import argparse
import dpdprops
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pint
import torch
from torch.optim import Adam

from rbc import (extract_dihedrals,
                 compute_area,
                 compute_volume,
                 compute_bending_energy,
                 compute_shear_energy)

def rescale_diameters(mesh, D):
    Dm = np.ptp(mesh.vertices[:,0])
    return D * Dm / D[0]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-csv', type=str, required=True, help="experimental data")
    parser.add_argument('--subdivisions', type=int, default=3, choices=[3, 4], help="resolution of the mesh")
    parser.add_argument('--sigma', type=float, default=0.05, help="measurements errors, in micron")
    parser.add_argument('--beta', type=float, default=1e-1, help="scale of beta factor, in 1/ambiant temperature energy units")
    args = parser.parse_args()

    lr = 5e-4
    num_epochs = 15001

    # RBC variables

    subdivisions = args.subdivisions
    mesh  = dpdprops.load_equilibrium_mesh(subdivisions=subdivisions)
    mesh0 = dpdprops.load_stress_free_mesh(subdivisions=subdivisions)

    print(f"nv = {len(mesh.vertices)}, nt = {len(mesh.faces)}")

    RA = dpdprops.equivalent_sphere_radius(area=mesh.area)

    dihedrals = extract_dihedrals(mesh.faces)
    faces = torch.from_numpy(mesh.faces)
    vertices0 = torch.from_numpy(mesh0.vertices).float()

    ureg = pint.UnitRegistry()
    params = dpdprops.JuelicherLimRBCDefaultParams(ureg)
    lscale = dpdprops.equivalent_sphere_radius(area=params.A0) / RA
    tscale = 1e-3 * ureg.second
    mscale = 1e-10 * ureg.g
    fscale = mscale * lscale / tscale**2
    p = params.get_params(length_scale=lscale,
                          time_scale=tscale,
                          mass_scale=mscale,
                          mesh=mesh)

    p.ka /= 1e3
    p.kv /= 1e3

    T_ = ureg.Quantity(20, ureg.degC).to('kelvin')
    kB_ = 1.380649e-23 * ureg.joule / ureg.kelvin
    kBT = float(kB_ * T_ / (lscale * fscale))

    sigma_ = args.sigma * ureg.micrometer
    beta = args.beta / kBT
    sigma = float(sigma_ / lscale)

    print(f"beta = {beta} ({(beta / lscale / fscale).to(1/ureg.joule)})")
    print(f"sigma = {sigma} ({(sigma * lscale).to(ureg.um)})")

    # Data
    df = pd.read_csv(args.data_csv)
    fmagn_ = np.array([float(f * ureg.piconewton / fscale) for f in df['Fext']])
    D0d = np.array([float(d * ureg.micrometer / lscale) for d in df['D0']])
    D1d = np.array([float(d * ureg.micrometer / lscale) for d in df['D1']])

    D0d = rescale_diameters(mesh, D0d)
    D1d = rescale_diameters(mesh, D1d)

    nv = len(mesh.vertices)
    ninputs = len(fmagn_)

    # force pulling on both sides of the mesh with magnitude fmagn and on nf vertices along x.
    nf = int(0.05 * nv)
    fmagn = fmagn_ / nf

    idx = np.argsort(mesh.vertices[:,0])
    idx_mf = torch.from_numpy(idx[:nf])
    idx_pf = torch.from_numpy(idx[-nf:])

    # Solution vector: vertices
    stride = 3 * nv
    y = torch.zeros(ninputs * stride)

    # initial guess
    for i in range(ninputs):
        y[i*stride:(i+1)*stride] = torch.from_numpy(mesh.vertices).float().flatten()

    def compute_internal_energy(vertices, p):

        A = compute_area(faces, vertices)
        V = compute_volume(faces, vertices)

        E_A = p.ka * (A - p.area)**2 / p.area
        E_V = p.kv * (V - p.volume)**2 / p.volume
        E_b = compute_bending_energy(faces,
                                     dihedrals,
                                     vertices,
                                     kb=p.bending_params.kb)
        E_s = compute_shear_energy(faces,
                                   vertices,
                                   vertices0,
                                   Ka=p.shear_params.ka,
                                   mu=p.shear_params.mu,
                                   a3=p.shear_params.a3,
                                   a4=p.shear_params.a4,
                                   b1=p.shear_params.b1,
                                   b2=p.shear_params.b2)
        return E_A + E_V + E_b + E_s

    def compute_beads_energy(vertices, f):
        # energy corresponding to a constant force on vertices.
        # it is thus -F * x, where x is position and F is the force magnitude
        x_r = vertices[idx_pf,0]
        x_l = vertices[idx_mf,0]
        Ebeads =  torch.sum(-f * x_r)
        Ebeads += torch.sum(+f * x_l)
        return Ebeads

    def compute_loss(y, p):
        loss = 0
        for i in range(ninputs):
            vertices = y[i*stride:(i+1)*stride].reshape((nv,3))
            energy = compute_internal_energy(vertices, p) + compute_beads_energy(vertices, fmagn[i])
            D0 = torch.max(vertices[:,1]) - torch.min(vertices[:,1])
            D1 = torch.max(vertices[:,0]) - torch.min(vertices[:,0])
            loss -= (D0 - D0d[i])**2 / (2 * sigma**2) + np.log(2 * np.pi * sigma**2) / 2
            loss -= (D1 - D1d[i])**2 / (2 * sigma**2) + np.log(2 * np.pi * sigma**2) / 2
            loss += beta * energy
        return loss

    def compute_neg_log_posterior(mu, initial_guess):
        params = p
        params.shear_params.mu = mu
        params.shear_params.ka = mu

        y = initial_guess.clone()
        y.requires_grad = True
        optim = Adam([y], lr=lr)
        patience = 10
        patience_count = 0
        best_loss = np.inf

        for epoch in range(num_epochs):
            optim.zero_grad()
            loss = compute_loss(y, params)
            loss.backward()
            optim.step()

            l = loss.item()

            if epoch % 1000 == 0:
                print(f"    epoch {epoch:06d} loss {loss:+.4e}")

            if l < best_loss:
                best_loss = l
                patience_count = 0
            else:
                patience_count += 1

            if patience_count >= patience:
                break

        return best_loss, y.detach().copy()



    mu0 = p.shear_params.mu

    mus = np.linspace(mu0/2, 2*mu0, 5)
    losses = []
    for mu in mus:
        loss, y = compute_neg_log_posterior(mu, y)
        losses.append(loss)
        print(f"mu {mu:.4e}, loss {loss:+.4e}")

    fig, ax = plt.subplots()
    ax.plot(mus, losses)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
