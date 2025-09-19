#!/usr/bin/env python

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import triangle as tr

def f(x, y):
    return x**3 + x**2 + 2 * y**2 + x * y + y**3

def lapl_exact(x, y):
    return 6 * (1 + x + y)

def triangle_areas(vertices, triangles):
    v0 = vertices[triangles[:,1]] - vertices[triangles[:,0]]
    v1 = vertices[triangles[:,2]] - vertices[triangles[:,0]]
    A = 0.5 * (v0[:,0]*v1[:,1] - v0[:,1]*v1[:,0])
    return A

def per_triangle_grad_scalar(vertices, triangles, f):
    """
    Returns G: (m,2) per-triangle gradients of scalar field f at vertices.
    """
    x = vertices[:,0]; y = vertices[:,1]
    i, j, k = triangles[:,0], triangles[:,1], triangles[:,2]
    A = triangle_areas(vertices, triangles)
    denom = 2.0 * A

    dphix_i =  (y[j] - y[k]) / denom
    dphiy_i =  (x[k] - x[j]) / denom
    dphix_j =  (y[k] - y[i]) / denom
    dphiy_j =  (x[i] - x[k]) / denom
    dphix_k =  (y[i] - y[j]) / denom
    dphiy_k =  (x[j] - x[i]) / denom

    Gx = f[i]*dphix_i + f[j]*dphix_j + f[k]*dphix_k
    Gy = f[i]*dphiy_i + f[j]*dphiy_j + f[k]*dphiy_k
    return Gx, Gy

def per_vertex_grad_scalar(vertices, triangles, f):
    Gtx, Gty = per_triangle_grad_scalar(vertices, triangles, f)
    A = np.abs(triangle_areas(vertices, triangles))
    n = vertices.shape[0]
    Gx = np.zeros(n)
    Gy = np.zeros(n)
    W  = np.zeros(n)
    for t, (a, b, c) in enumerate(triangles):
        w = A[t]
        gradx = Gtx[t]
        grady = Gty[t]
        for v in (a, b, c):
            Gx[v] += w * gradx
            Gy[v] += w * grady
            W[v]  += w
    W = np.maximum(W, 1e-16)
    Gx /= W
    Gy /= W
    return Gx, Gy

def vertex_areas(vertices, triangles):
    p = vertices
    tri = triangles
    a = p[tri[:,1]] - p[tri[:,0]]
    b = p[tri[:,2]] - p[tri[:,0]]
    Atri = 0.5 * np.abs(a[:,0]*b[:,1] - a[:,1]*b[:,0])
    nverts = len(vertices)
    A = np.zeros(nverts)
    np.add.at(A, tri[:,0], Atri/3)
    np.add.at(A, tri[:,1], Atri/3)
    np.add.at(A, tri[:,2], Atri/3)
    return A

def laplacian_f(vertices, triangles, fvals):
    Gx, Gy = per_vertex_grad_scalar(vertices, triangles, fvals)
    Gxx, Gxy = per_vertex_grad_scalar(vertices, triangles, Gx)
    Gyx, Gyy = per_vertex_grad_scalar(vertices, triangles, Gy)
    return Gxx + Gyy


def ring_segments(offset, n):
    idx = np.arange(offset, offset + n)
    return np.column_stack([idx, np.roll(idx, -1)])

def test():
    L = 2
    poly = {
        'vertices': np.array([[-L/2, -L/2],
                              [-L/2, +L/2],
                              [+L/2, +L/2],
                              [+L/2, -L/2],
                              ]),
        'segments': ring_segments(0, 4)
    }

    # Flags:
    #  q : quality mesh (default min angle ~20°; use q30 for 30°)
    #  a : max triangle area (e.g., a0.1)
    #  p : PSLG (respect segments)
    #  D : produce Delaunay (optional)
    area = 0.02
    mesh = tr.triangulate(poly, f"q30a{area:.10f}")
    vertices = mesh["vertices"]
    triangles = mesh["triangles"]
    x = vertices[:,0]
    y = vertices[:,1]

    lapl_f_exact = lapl_exact(x, y)
    lapl_f = laplacian_f(vertices, triangles, f(x, y))

    fig, ax = plt.subplots()
    ax.plot(lapl_f_exact, lapl_f, 'ok')
    ax.set_aspect('equal')
    ax.plot([-10, 20], [-10, 20], '--k')
    plt.show()

    norm = mcolors.Normalize(
        vmin=min(lapl_f_exact.min(), lapl_f.min()),
        vmax=max(lapl_f_exact.max(), lapl_f.max())
    )
    cmap = plt.cm.viridis

    fig, axes = plt.subplots(figsize=(8,4), ncols=2)
    ax = axes[0]
    ax.triplot(x, y, triangles, c='k', lw=0.1)
    ax.scatter(x, y, c=lapl_f_exact, norm=norm, cmap=cmap)
    ax.set_xlim(-L/2, L/2)
    ax.set_ylim(-L/2, L/2)
    ax.set_aspect('equal')
    ax.axis("off")

    ax = axes[1]
    ax.triplot(x, y, triangles, c='k', lw=0.1)
    ax.scatter(x, y, c=lapl_f, norm=norm, cmap=cmap)
    ax.set_xlim(-L/2, L/2)
    ax.set_ylim(-L/2, L/2)
    ax.set_aspect('equal')
    ax.axis("off")

    plt.tight_layout()
    plt.show()


def convergence():
    L = 2

    poly = {
        'vertices': np.array([[-L/2, -L/2],
                              [-L/2, +L/2],
                              [+L/2, +L/2],
                              [+L/2, -L/2],
                              ]),
        'segments': ring_segments(0, 4)
    }

    def err(h):
        area = h**2 / 2
        mesh = tr.triangulate(poly, f"pq30a{area:.10f}")
        vertices = mesh["vertices"]
        triangles = mesh["triangles"]
        x = vertices[:,0]
        y = vertices[:,1]

        lapl_f_exact = lapl_exact(x, y)
        lapl_f = laplacian_f(vertices, triangles, f(x, y))
        A = vertex_areas(vertices, triangles)
        err2 = A * (lapl_f - lapl_f_exact)**2

        return np.sqrt(np.sum(err2) / np.sum(A))

    h = 1 / 2**np.arange(1, 7)
    errors = [err(h) for h in h]

    fig, ax = plt.subplots()
    ax.plot(1/h, errors, '-o')
    ax.plot(1/h, h**0.5, '--k')
    ax.set_xlabel(r"$1/h$")
    ax.set_ylabel("error")
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.show()

def main():
    test()
    convergence()

if __name__ == '__main__':
    main()
