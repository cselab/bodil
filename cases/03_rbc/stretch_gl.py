#!/usr/bin/env python3

import argparse
import dpdprops
import numpy as np
import pint
import torch
from torch.optim import Adam

from rbc import (extract_dihedrals,
                 compute_area,
                 compute_volume,
                 compute_bending_energy,
                 compute_shear_energy)

from graph_laplacian import compute_graph_laplacian_basis

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kmax', type=int, default=100, help='number of basis functions to use.')
    parser.add_argument('--subdivisions', type=int, default=3, choices=[3,4], help='Mesh resolution.')
    args = parser.parse_args()

    kmax = args.kmax

    subdivisions = args.subdivisions
    mesh  = dpdprops.load_equilibrium_mesh(subdivisions=subdivisions)
    mesh0 = dpdprops.load_stress_free_mesh(subdivisions=subdivisions)

    phi = compute_graph_laplacian_basis(mesh, k_max=kmax)
    phi_ = torch.from_numpy(phi).float()

    RA = dpdprops.equivalent_sphere_radius(area=mesh.area)

    dihedrals = extract_dihedrals(mesh.faces)
    faces = torch.from_numpy(mesh.faces)
    vertices = torch.from_numpy(mesh.vertices).float()
    vertices0 = torch.from_numpy(mesh0.vertices).float()
    c = phi_.T @ vertices

    c.requires_grad = True

    ureg = pint.UnitRegistry()
    params = dpdprops.JuelicherLimRBCDefaultParams(ureg)
    lscale = dpdprops.equivalent_sphere_radius(area=params.A0) / RA
    tscale = 1e-3 * ureg.second
    mscale = 1e-10 * ureg.g
    p = params.get_params(length_scale=lscale,
                          time_scale=tscale,
                          mass_scale=mscale,
                          mesh=mesh)

    p.ka /= 1e3
    p.kv /= 1e3

    # force pulling on both sides of the mesh with magnitude fmagn and on nf vertices along x.
    nf = int(0.05 * len(vertices))
    fmagn_ = 100 * ureg.piconewton
    fscale = mscale * lscale / tscale**2
    fmagn = float(fmagn_ / fscale)
    fmagn /= nf

    idx = np.argsort(mesh.vertices[:,0])
    idx_mf = torch.from_numpy(idx[:nf])
    idx_pf = torch.from_numpy(idx[-nf:])


    def compute_internal_energy(c):
        vertices = phi_ @ c
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

    def compute_beads_energy(c):
        vertices = phi_ @ c
        # energy corresponding to a constant force on vertices.
        # it is thus -F * x, where x is position and F is the force magnitude
        x_r = vertices[idx_pf,0]
        x_l = vertices[idx_mf,0]
        Ebeads =  torch.sum(-fmagn * x_r)
        Ebeads += torch.sum(+fmagn * x_l)
        return Ebeads

    def compute_loss(c):
        energy = compute_internal_energy(c) + compute_beads_energy(c)
        return energy

    optim = Adam([c], lr=5e-3)

    dump_id = 0
    for epoch in range(5001):
        optim.zero_grad()
        loss = compute_loss(c)
        loss.backward()
        optim.step()

        if epoch % 500 == 0:
            l = loss.item()
            print(f"epoch {epoch:06d} loss {l:.4e}")

            mesh.vertices = (phi_ @ c).detach().numpy()
            mesh.export(f"stretch-gl-{dump_id:06d}.ply")
            dump_id += 1


if __name__ == '__main__':
    main()
