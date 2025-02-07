#!/usr/bin/env python3

import argparse
import dpdprops
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pint
from scipy.stats import multivariate_normal
import torch
from torch.optim import Adam

from rbc import (extract_dihedrals,
                 compute_area,
                 compute_volume,
                 compute_bending_energy,
                 compute_shear_energy)

from graph_laplacian import compute_graph_laplacian_basis

def rescale_diameters(mesh, D):
    Dm = np.ptp(mesh.vertices[:,0])
    return D * Dm / D[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-csv', type=str, required=True, help="experimental data")
    parser.add_argument('--subdivisions', type=int, default=3, choices=[3, 4], help="resolution of the mesh")
    parser.add_argument('--sigma', type=float, default=0.1, help="measurements errors, in micron")
    parser.add_argument('--kmax', type=int, default=100, help='number of basis functions to use.')
    args = parser.parse_args()

    lr = 5e-4
    num_epochs = 15001
    stats_every = 500
    num_samples = 1000
    seed = 923868
    kmax = args.kmax
    rng = np.random.default_rng(seed)

    # RBC variables

    subdivisions = args.subdivisions
    mesh  = dpdprops.load_equilibrium_mesh(subdivisions=subdivisions)
    mesh0 = dpdprops.load_stress_free_mesh(subdivisions=subdivisions)

    phi = compute_graph_laplacian_basis(mesh, k_max=kmax)
    phi_ = torch.from_numpy(phi).float()

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
    beta = 1 / kBT
    sigma = float(sigma_ / lscale)

    print(f"beta = {beta}")
    print(f"sigma = {sigma}")

    # Data
    df = pd.read_csv(args.data_csv)
    fmagn = np.array([float(f * ureg.piconewton / fscale) for f in df['Fext']])
    D0d = np.array([float(d * ureg.micrometer / lscale) for d in df['D0']])
    D1d = np.array([float(d * ureg.micrometer / lscale) for d in df['D1']])

    D0d = rescale_diameters(mesh, D0d)
    D1d = rescale_diameters(mesh, D1d)

    nv = len(mesh.vertices)
    ninputs = len(fmagn)

    # force pulling on both sides of the mesh with magnitude fmagn and on nf vertices along x.
    nf = int(0.05 * nv)
    fmagn /= nf

    idx = np.argsort(mesh.vertices[:,0])
    idx_mf = torch.from_numpy(idx[:nf])
    idx_pf = torch.from_numpy(idx[-nf:])


    # Solution vector: vertices projected on basis functions
    stride = 3 * kmax
    y = torch.zeros(ninputs * stride)

    # initial guess
    for i in range(ninputs):
        y[i*stride:(i+1)*stride] = (phi_.T @ torch.from_numpy(mesh.vertices).float()).flatten()

    y.requires_grad = True

    def compute_internal_energy(vertices):

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

    def compute_attachment_energy(vertices):
        cm = torch.mean(vertices, dim=0)
        return torch.sum(cm**2)

    def compute_neg_posterior(y):
        energy = 0
        log_likelihood = 0
        for i in range(ninputs):
            vertices = phi_ @ y[i*stride:(i+1)*stride].reshape((kmax,3))
            energy += compute_internal_energy(vertices) + compute_beads_energy(vertices, fmagn[i])
            energy += compute_attachment_energy(vertices)
            D0 = torch.max(vertices[:,1]) - torch.min(vertices[:,1])
            D1 = torch.max(vertices[:,0]) - torch.min(vertices[:,0])
            log_likelihood -= (D0 - D0d[i])**2 / (2 * sigma**2) + np.log(2 * np.pi * sigma**2) / 2
            log_likelihood -= (D1 - D1d[i])**2 / (2 * sigma**2) + np.log(2 * np.pi * sigma**2) / 2

        nlp = beta * energy - log_likelihood
        return nlp

    optim = Adam([y], lr=lr)

    for epoch in range(num_epochs):
        optim.zero_grad()
        loss = compute_neg_posterior(y)
        loss.backward()
        optim.step()

        if epoch % stats_every == 0:
            l = loss.item()
            print(f"epoch {epoch:06d} loss {l:.4e}")


    # Laplace approximation
    print(f"Computing Hessian (size {len(y)})...")
    H = torch.autograd.functional.hessian(compute_neg_posterior, y, create_graph=True)
    H = H.detach().numpy()
    y = y.detach().numpy()


    print(f"Generating {num_samples} samples...")
    eigvals, eigvecs = np.linalg.eigh(H)
    # eigvals = np.maximum(1, eigvals) # TODO this is a hack.
    eigvals = np.where(eigvals < 1e2, 1e6, eigvals) # TODO this is a hack.
    samples = np.empty((len(y), num_samples))

    for k in range(num_samples):
        z = rng.normal(0, 1/np.sqrt(eigvals), len(y))
        samples[:,k] = y + eigvecs @ z

    print("Done.")


    D0 = []
    D1 = []
    D0_samples = []
    D1_samples = []
    for i in range(ninputs):
        vertices = phi @ y[i*stride:(i+1)*stride].reshape((kmax,3))
        D0.append(np.max(vertices[:,1]) - np.min(vertices[:,1]))
        D1.append(np.max(vertices[:,0]) - np.min(vertices[:,0]))

        vertices_samples = np.einsum('ij,jlk->ilk', phi, samples[i*stride:(i+1)*stride,:].reshape((kmax,3,num_samples)))
        D0_samples.append(np.max(vertices_samples[:,1,:], axis=-1) - np.min(vertices_samples[:,1,:], axis=-1))
        D1_samples.append(np.max(vertices_samples[:,0,:], axis=-1) - np.min(vertices_samples[:,0,:], axis=-1))

        mesh.vertices = vertices
        mesh.export(f"stretch-{i:06d}.ply")

        path = f"samples_{i}"
        os.makedirs(path, exist_ok=True)

        for k in range(num_samples):
            mesh.vertices = vertices_samples[:,:,k]
            mesh.export(os.path.join(path, f"sample-{k:06d}.ply"))


    D0_samples = np.array(D0_samples)
    D1_samples = np.array(D1_samples)

    D0_lo = np.quantile(D0_samples, q=0.05, axis=1)
    D0_hi = np.quantile(D0_samples, q=0.95, axis=1)

    D1_lo = np.quantile(D1_samples, q=0.05, axis=1)
    D1_hi = np.quantile(D1_samples, q=0.95, axis=1)

    fig, ax = plt.subplots()

    # ax.plot(fmagn, D0, color='C0')
    # ax.plot(fmagn, D1, color='C0')

    ax.errorbar(fmagn, D0, np.vstack((D0_lo, D0_hi)), color='C0', capsize=2, fmt='o')
    ax.errorbar(fmagn, D1, np.vstack((D1_lo, D1_hi)), color='C0', capsize=2, fmt='o')

    ax.plot(fmagn, D0d, color='C0', ls='none', marker='+')
    ax.plot(fmagn, D1d, color='C0', ls='none', marker='+')

    ax.set_xlabel(r"$F_{ext}$ (pN)")
    ax.set_ylabel(r"$D$ ($\mu$m)")
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
