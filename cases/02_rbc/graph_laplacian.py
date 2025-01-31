#!/usr/bin/env python3

import numpy as np
import trimesh
import scipy

def compute_graph_laplacian_basis(mesh, k_max=None):
    nv = len(mesh.vertices)

    if k_max is None:
        k_max = nv - 2

    _, edges = trimesh.graph.face_adjacency(faces=mesh.faces, return_edges=True)
    ne = len(edges)
    A = trimesh.graph.edges_to_coo(edges, count=nv, data=np.ones(ne))
    A += A.T

    degrees = np.array(mesh.vertex_degree)
    D = scipy.sparse.diags(degrees)

    L = D - A
    lam, phi = scipy.sparse.linalg.eigsh(L, k=k_max, which='SA')
    return phi
