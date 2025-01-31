#!/usr/bin/env python3

import torch

def extract_dihedrals(faces):
    r"""
    Find dihedrals from face connectivity information.
    Assume a closed triangle mesh.
    The order of the indices are as follow:

    (b, c, a, d)

        a
       /|\
      / | \
     d  |  c
      \ | /
       \|/
        b

    Arguments:
        faces: Face connectivity of the triangle mesh.

    Returns:
        The dihedral informtions (one entry of 4 indices per dihedral)
    """

    edges_to_faces = {}

    for faceid, f in enumerate(faces):
        for i in range(3):
            edge = (f[i], f[(i+1)%3])
            edges_to_faces[edge] = faceid

    dihedrals = []

    for faceid, f0 in enumerate(faces):
        for i in range(3):
            a = f0[i]
            b = f0[(i+1)%3]
            c = f0[(i+2)%3]
            edge = (b, a)
            otherfaceid = edges_to_faces[edge]
            f1 = faces[otherfaceid]
            d = [v for v in f1 if v != a and v != b]
            assert len(d) == 1
            d = d[0]
            dihedrals.append([b, c, a, d])

    return torch.tensor(dihedrals)

def _compute_dihedral_normals(dihedrals, vertices):
    b = vertices[dihedrals[:,0],:]
    c = vertices[dihedrals[:,1],:]
    a = vertices[dihedrals[:,2],:]
    d = vertices[dihedrals[:,3],:]
    ab = b-a
    n0 = torch.linalg.cross(ab, c-a, dim=1)
    n1 = torch.linalg.cross(ab, a-d, dim=1)
    n0 /= torch.linalg.norm(n0, dim=1)[:,None]
    n1 /= torch.linalg.norm(n1, dim=1)[:,None]
    return n0, n1

def _compute_triangle_areas(faces, vertices):
    a = vertices[faces[:,0],:]
    b = vertices[faces[:,1],:]
    c = vertices[faces[:,2],:]
    n = torch.linalg.cross(b - a, c - a, dim=1)
    return 0.5 * torch.linalg.norm(n, dim=1)


def compute_vertex_mean_curvatures(faces,
                                   dihedrals,
                                   vertices,
                                   return_vertex_areas=False):
    """
    Compute the mean curvature of each vertex.

    Arguments:
        faces: connectivity of the mesh.
        dihedrals: return of extract_dihedrals()
        vertices: positions of the mesh vertices.
        return_vertex_areas: if True, also return the area associated to each vertex.

    Return:
        The mean curvature of each vertex.
        Optionally, also returns the vertex areas.
    """
    nv = len(vertices)

    faces_areas = _compute_triangle_areas(faces, vertices)

    b = vertices[dihedrals[:,0],:]
    c = vertices[dihedrals[:,1],:]
    a = vertices[dihedrals[:,2],:]
    d = vertices[dihedrals[:,3],:]

    ab = b-a
    n0 = torch.linalg.cross(ab, c-a, dim=1)
    n1 = torch.linalg.cross(ab, a-d, dim=1)

    n0n1 = torch.linalg.cross(n0, n1, dim=1)
    arg = torch.linalg.norm(n0n1, dim=1) / (torch.linalg.norm(n0, dim=1) * torch.linalg.norm(n1, dim=1))
    theta = torch.asin(arg)
    l = torch.linalg.norm(ab, dim=1)
    ltheta = l * theta

    vertex_areas = torch.zeros(nv)
    vertex_areas.index_add_(dim=0, index=faces[:,0], source=faces_areas)
    vertex_areas.index_add_(dim=0, index=faces[:,1], source=faces_areas)
    vertex_areas.index_add_(dim=0, index=faces[:,2], source=faces_areas)
    vertex_areas /= 3

    vertex_mean_curvatures = torch.zeros(nv)
    vertex_mean_curvatures.index_add_(dim=0, index=dihedrals[:,2], source=ltheta)
    vertex_mean_curvatures.index_add_(dim=0, index=dihedrals[:,0], source=ltheta)
    vertex_mean_curvatures /= 8 * vertex_areas

    if return_vertex_areas:
        return vertex_mean_curvatures, vertex_areas
    else:
        return vertex_mean_curvatures


def compute_bending_energy(faces,
                           dihedrals,
                           vertices,
                           kb,
                           H0=0,
                           kade=0,
                           deltaA0=0):
    """
    Compute the bending energy using the Juelicher model.

    Arguments:
        faces: connectivity of the mesh.
        dihedrals: return of extract_dihedrals()
        vertices: positions of the mesh.
        kb: bending energy coefficient for the Helfrich term
        H0: spontaneous mean curvature
        kade: bending energy coefficient for the ADE term (alpha * kb * pi / D**2 in Juelicher1997 or Bian2020)
        deltaA0: equilibrium area difference
    Return:
        The total bending energy of the mesh.
    """

    vertex_mean_curvatures, vertex_areas = compute_vertex_mean_curvatures(faces=faces,
                                                                          dihedrals=dihedrals,
                                                                          vertices=vertices,
                                                                          return_vertex_areas=True)

    # Helfrich energy
    EH = 2 * kb * torch.sum((vertex_mean_curvatures - H0)**2 * vertex_areas)

    # ADE energy
    deltaA = torch.sum(vertex_mean_curvatures * vertex_areas)
    A = torch.sum(vertex_areas)
    EADE = kade / (2 * A) * (deltaA - deltaA0)**2
    return EH + EADE

def compute_shear_energy(faces,
                         vertices,
                         vertices0,
                         Ka, mu,
                         a3=-2.0, a4=8.0, b1=0.7, b2=1.84):
    """
    Compute the shear energy with respect to the SFS using the Lim model.

    Arguments:
        faces: connectivity of the mesh.
        vertices: positions of the mesh vertices.
        vertices0: positions of the unstressed mesh vertices.
        Ka: area dilation coefficient.
        mu: shear modulus
        a3, a4, b1, b2: non linear coefficients.

    Return:
        The total shear energy of the mesh.
    """

    v1 = vertices[faces[:,0]]
    v2 = vertices[faces[:,1]]
    v3 = vertices[faces[:,2]]

    u1 = vertices0[faces[:,0]]
    u2 = vertices0[faces[:,1]]
    u3 = vertices0[faces[:,2]]

    y12 = u2 - u1
    y13 = u3 - u1
    eq_area = 0.5 * torch.linalg.norm(torch.linalg.cross(y12, y13, dim=1), dim=1)
    eq_dotp = torch.sum(y12*y13, dim=1)

    x12 = v2 - v1
    x13 = v3 - v1

    area = 0.5 * torch.linalg.norm(torch.linalg.cross(x12, x13, dim=1), dim=1)
    area_inv = 1.0 / area
    area0_inv = 1.0 / eq_area

    alpha = area * area0_inv - 1.0

    e0sq_A = torch.sum(x12*x12, dim=1) * area_inv
    e1sq_A = torch.sum(x13*x13, dim=1) * area_inv

    e0sq_A0 = torch.sum(y12*y12, dim=1) * area0_inv
    e1sq_A0 = torch.sum(y13*y13, dim=1) * area0_inv

    dotp = torch.sum(x12*x13, dim=1)

    dot_4A = 0.25 * eq_dotp * area0_inv
    mixed_v = 0.125 * (e0sq_A0 * e1sq_A + e1sq_A0 * e0sq_A)

    beta = mixed_v - dot_4A * dotp * area_inv - 1.0

    return 0.5 * Ka * torch.sum((alpha**2 + a3 * alpha**3 + a4 * alpha**4) * eq_area) + \
        mu * torch.sum((beta + b1*alpha*beta + b2*beta**2) * eq_area)

def compute_area(faces, vertices):
    areas = _compute_triangle_areas(faces, vertices)
    return torch.sum(areas)

def compute_volume(faces, vertices):
    a = vertices[faces[:,0],:]
    b = vertices[faces[:,1],:]
    c = vertices[faces[:,2],:]
    n = torch.linalg.cross(b-a, c-a, axis=1)
    return torch.sum(a * n) / 6
