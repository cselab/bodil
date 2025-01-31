#!/usr/bin/env python3

import argparse
import dpdprops
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
    parser.add_argument('--kmax', type=int, default=50, help='number of basis functions to use.')
    args = parser.parse_args()

    kmax = args.kmax

    mesh  = dpdprops.load_equilibrium_mesh(subdivisions=4)
    mesh0 = dpdprops.load_stress_free_mesh(subdivisions=4)

    RA = dpdprops.equivalent_sphere_radius(area=mesh.area)

    dihedrals = extract_dihedrals(mesh.faces)
    faces = torch.from_numpy(mesh.faces)
    vertices = torch.from_numpy(mesh.vertices).float()
    vertices0 = torch.from_numpy(mesh0.vertices).float()

    phi = torch.from_numpy(compute_graph_laplacian_basis(mesh, k_max=kmax)).float()
    c = phi.T @ vertices

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

    def compute_energy(c):
        vertices = phi @ c
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


    optim = Adam([c], lr=1e-3)

    dump_id = 0
    for epoch in range(10000):
        optim.zero_grad()
        loss = compute_energy(c)
        loss.backward()
        optim.step()

        if epoch % 1000 == 0:
            l = loss.item()
            print(f"epoch {epoch:06d} loss {l:.4e}")

            mesh.vertices = (phi @ c.detach()).numpy()
            mesh.export(f"crbc-{dump_id:06d}.ply")
            dump_id += 1


if __name__ == '__main__':
    main()
