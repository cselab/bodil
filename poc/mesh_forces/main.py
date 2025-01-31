#!/usr/bin/env python3

import numpy as np
import torch
from torch.optim import Adam
import trimesh


def _compute_triangle_areas(faces, vertices):
    a = vertices[faces[:,0],:]
    b = vertices[faces[:,1],:]
    c = vertices[faces[:,2],:]
    n = torch.linalg.cross(b - a, c - a, dim=1)
    return 0.5 * torch.linalg.norm(n, dim=1)

def compute_area(faces, vertices):
    areas = _compute_triangle_areas(faces, vertices)
    return torch.sum(areas)

def compute_volume(faces, vertices):
    a = vertices[faces[:,0],:]
    b = vertices[faces[:,1],:]
    c = vertices[faces[:,2],:]
    n = torch.linalg.cross(b-a, c-a, axis=1)
    return torch.sum(a * n) / 6

def main():
    mesh = trimesh.creation.icosphere(radius=1, subdivisions=2)
    faces = torch.from_numpy(mesh.faces)
    vertices = torch.from_numpy(mesh.vertices)
    vertices.requires_grad=True


    A0 = mesh.area
    a0 = A0 / len(faces)
    V0 = mesh.volume * 0.7

    # force pulling on both sides of the mesh with magnitude fmagn and on nf vertices along x.
    nf = 5
    fmagn = 1.0

    idx = np.argsort(mesh.vertices[:,0])
    idx_mf = torch.from_numpy(idx[:nf])
    idx_pf = torch.from_numpy(idx[-nf:])


    def forces_residuals(vertices):
        areas = _compute_triangle_areas(faces, vertices)
        A = torch.sum(areas)
        V = compute_area(faces, vertices)
        energy = 10 * ((A - A0) / A0)**2 + 10 * ((V - V0) / V0)**2
        energy += torch.sum(((areas - a0) / a0)**2)
        forces = torch.autograd.grad(-energy, inputs=vertices, create_graph=True)[0]
        forces[idx_mf, 0] -= fmagn
        forces[idx_pf, 0] += fmagn
        return forces


    optim = Adam([vertices], lr=0.005)

    for epoch in range(5000):
        optim.zero_grad()
        res = forces_residuals(vertices)
        loss = torch.mean(res**2)
        loss.backward()
        optim.step()

        if epoch % 100 == 0:
            l = loss.item()
            print(f"epoch {epoch:06d} loss {l:.4e}")

    mesh.vertices = vertices.detach().numpy()
    mesh.export("output.ply")


if __name__ == '__main__':
    main()
