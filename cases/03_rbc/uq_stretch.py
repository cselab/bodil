#!/usr/bin/env python3

import argparse
import dpdprops
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pint
import torch
from torch.optim import Adam

from rbc import (extract_dihedrals,
                 compute_area,
                 compute_volume,
                 compute_bending_energy,
                 compute_shear_energy)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-csv', type=str, help="experimental data")
    parser.add_argument('--subdivisions', type=int, default=3, choices=[3, 4], help="resolution of the mesh")
    parser.add_argument('--sigma', type=float, default=0.1, help="measurements errors, in micron")
    args = parser.parse_args()

    lr = 1e-4
    num_epochs = 10000
    stats_every = 1000

    # RBC variables

    subdivisions = args.subdivisions
    mesh  = dpdprops.load_equilibrium_mesh(subdivisions=subdivisions)
    mesh0 = dpdprops.load_stress_free_mesh(subdivisions=subdivisions)

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
    beta = 1/kBT
    sigma = float(sigma_ / lscale)

    print(f"beta = {beta}")
    print(f"sigma = {sigma}")

    # Data
    df = pd.read_csv(args.data_csv)
    fmagn = np.array([float(f * ureg.piconewton / fscale) for f in df['Fext']])
    D0d = np.array([float(d * ureg.micrometer / lscale) for d in df['D0']])
    D1d = np.array([float(d * ureg.micrometer / lscale) for d in df['D1']])

    nv = len(mesh.vertices)
    ninputs = len(fmagn)

    # force pulling on both sides of the mesh with magnitude fmagn and on nf vertices along x.
    nf = int(0.05 * nv)
    fmagn /= nf

    idx = np.argsort(mesh.vertices[:,0])
    idx_mf = torch.from_numpy(idx[:nf])
    idx_pf = torch.from_numpy(idx[-nf:])


    # Solution vector: vertices, mu.
    y = torch.zeros(3 * nv * ninputs + 1)

    # initial guess
    stride = 3 * nv
    for i in range(ninputs):
        y[i*stride:(i+1)*stride] = torch.from_numpy(mesh.vertices.flatten()).float()
    y[ninputs*stride + 0] = p.shear_params.mu

    y.requires_grad = True

    def compute_internal_energy(vertices, mu):

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
                                   Ka=mu,
                                   mu=mu,
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

    def compute_neg_posterior(y):
        energy = 0
        log_likelihood = 0
        mu = y[ninputs * stride + 0]
        for i in range(ninputs):
            vertices = y[i*stride:(i+1)*stride].reshape((nv,3))
            energy += compute_internal_energy(vertices, mu) + compute_beads_energy(vertices, fmagn[i])
            D0 = torch.max(vertices[:,1]) - torch.min(vertices[:,1])
            D1 = torch.max(vertices[:,0]) - torch.min(vertices[:,0])
            log_likelihood -= (D0 - D0d[i])**2 / (2 * sigma**2) - np.log(2 * np.pi * sigma**2) / 2
            log_likelihood -= (D1 - D1d[i])**2 / (2 * sigma**2) - np.log(2 * np.pi * sigma**2) / 2

        nlp = beta * energy - log_likelihood
        return nlp

    optim = Adam([y], lr=lr)

    dump_id = 0
    for epoch in range(num_epochs):
        optim.zero_grad()
        loss = compute_neg_posterior(y)
        loss.backward()
        optim.step()

        if epoch % stats_every == 0:
            l = loss.item()
            mu = y[-1].item()
            print(f"epoch {epoch:06d} loss {l:.4e} mu {mu:.4e}")

            #mesh.vertices = vertices.detach().numpy()
            #mesh.export(f"stretch-{dump_id:06d}.ply")
            dump_id += 1


    D0 = []
    D1 = []
    for i in range(ninputs):
        vertices = y[i*stride:(i+1)*stride].reshape((nv,3)).detach().numpy()
        D0.append(np.max(vertices[:,1]) - np.min(vertices[:,1]))
        D1.append(np.max(vertices[:,0]) - np.min(vertices[:,0]))

        mesh.vertices = vertices
        mesh.export(f"stretch-{i:06d}.ply")

    fig, ax = plt.subplots()

    ax.plot(fmagn, D0, color='C0')
    ax.plot(fmagn, D1, color='C0')

    ax.plot(fmagn, D0d, color='C0', ls='none', marker='+')
    ax.plot(fmagn, D1d, color='C0', ls='none', marker='+')

    ax.set_xlabel(r"$F_{ext}$ (pN)")
    ax.set_ylabel(r"$D$ ($\mu$m)")
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
