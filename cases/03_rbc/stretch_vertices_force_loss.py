#!/usr/bin/env python3

import dpdprops
import matplotlib.pyplot as plt
import numpy as np
import pint
import torch
from torch.optim import Adam

from rbc import (extract_dihedrals,
                 compute_vertex_areas,
                 compute_area,
                 compute_volume,
                 compute_bending_energy,
                 compute_shear_energy)

def main():
    dtype = torch.float64

    subdivisions = 3
    mesh  = dpdprops.load_equilibrium_mesh(subdivisions=subdivisions)
    mesh0 = dpdprops.load_stress_free_mesh(subdivisions=subdivisions)

    RA = dpdprops.equivalent_sphere_radius(area=mesh.area)

    dihedrals = extract_dihedrals(mesh.faces)
    faces = torch.from_numpy(mesh.faces)
    vertices = torch.from_numpy(mesh.vertices).to(dtype)
    vertices0 = torch.from_numpy(mesh0.vertices).to(dtype)
    vertices.requires_grad = True

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


    def compute_internal_energy():
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

    def compute_beads_energy():
        # energy corresponding to a constant force on vertices.
        # it is thus -F * x, where x is position and F is the force magnitude
        x_r = vertices[idx_pf,0]
        x_l = vertices[idx_mf,0]
        Ebeads =  torch.sum(-fmagn * x_r)
        Ebeads += torch.sum(+fmagn * x_l)
        return Ebeads

    def compute_loss():

        energy = compute_internal_energy() + compute_beads_energy()
        forces = torch.autograd.grad(-energy, inputs=vertices,
                                     create_graph=True, materialize_grads=True)[0]
        #areas = compute_vertex_areas(faces, vertices)
        loss = torch.sum(forces**2)
        return energy, loss

    optim = Adam([vertices], lr=1e-3)

    epochs = list(range(100001))
    flosses = []
    energies = []
    dump_id = 0
    for epoch in epochs:
        optim.zero_grad()
        energy, loss = compute_loss()
        #energy.backward()
        loss.backward()
        optim.step()

        flosses.append(loss.item())
        energies.append(energy.item())

        if epoch % 1000 == 0:
            e = energy.item()
            l = loss.item()
            print(f"epoch {epoch:06d} energy {e:.4e} loss {l:.4e}")

            mesh.vertices = vertices.detach().numpy()
            mesh.export(f"stretch-{dump_id:06d}.ply")
            dump_id += 1

    fig, axes = plt.subplots(ncols=2, figsize=(9.8, 4.8))
    ax = axes[0]
    ax.plot(epochs, flosses)
    ax.set_xlabel('epoch')
    ax.set_ylabel('forces loss')
    ax.set_yscale('log')
    ax = axes[1]
    ax.plot(epochs, energies)
    ax.set_xlabel('epoch')
    ax.set_ylabel('energy')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
