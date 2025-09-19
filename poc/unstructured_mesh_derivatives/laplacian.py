#!/usr/bin/env python

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import triangle as tr

def f(x, y):
    return x**3 + x**2 + 2 * y**2 + x * y + y**3

def grad_f_exact(x, y):
    return 3 * x**2 + 2 * x + y, 4 * y + x + 3 * y**2

def lapl_exact(x, y):
    return 6 * (1 + x + y)

def cotan_stiffness(vertices, triangles):
    n = vertices.shape[0]
    S = np.zeros((n, n))

    def cot_at(a, b, c):
        u = vertices[b] - vertices[a]
        v = vertices[c] - vertices[a]
        dot = np.dot(u, v)
        cross = u[0]*v[1] - u[1]*v[0]
        return dot / max(1e-16, abs(cross))

    for (i, j, k) in triangles:
        wi = 0.5 * cot_at(i, j, k)  # opposite edge (j,k)
        wj = 0.5 * cot_at(j, k, i)  # opposite edge (k,i)
        wk = 0.5 * cot_at(k, i, j)  # opposite edge (i,j)

        # edge (j,k), weight wi
        S[j, j] += wi; S[k, k] += wi
        S[j, k] -= wi; S[k, j] -= wi
        # edge (k,i), weight wj
        S[k, k] += wj; S[i, i] += wj
        S[k, i] -= wj; S[i, k] -= wj
        # edge (i,j), weight wk
        S[i, i] += wk; S[j, j] += wk
        S[i, j] -= wk; S[j, i] -= wk
    return S

def vertex_areas(vertices, triangles):
    p = vertices; tri = triangles
    a = p[tri[:,1]] - p[tri[:,0]]
    b = p[tri[:,2]] - p[tri[:,0]]
    Atri = 0.5 * np.abs(a[:,0]*b[:,1] - a[:,1]*b[:,0])
    n = len(vertices)
    A = np.zeros(n)
    np.add.at(A, tri[:,0], Atri/3)
    np.add.at(A, tri[:,1], Atri/3)
    np.add.at(A, tri[:,2], Atri/3)
    return A

def boundary_edges(vertices, triangles):
    # return list of (i,j,t) where edge (i->j) is a boundary edge of triangle t,
    # oriented as it appears in triangle t (needed to pick outward normal)
    F = triangles
    edges = {}
    for t, (i,j,k) in enumerate(F):
        for a,b in [(i,j),(j,k),(k,i)]:
            key = (min(a,b), max(a,b))
            if key in edges: edges[key].append((t,a,b))
            else:            edges[key] = [(t,a,b)]
    bnd = []
    for key, lst in edges.items():
        if len(lst) == 1:
            t,a,b = lst[0]     # oriented as in triangle t
            bnd.append((a,b,t))
    return bnd

def boundary_flux_load(vertices, triangles, grad_f_exact):
    """
    Assemble b_i = ∮ φ_i * (∂f/∂n) ds over boundary using midpoint rule on each boundary edge.
    """
    b = np.zeros(len(vertices))
    for (a,bv,t) in boundary_edges(vertices, triangles):
        i, j = a, bv
        vi, vj = vertices[i], vertices[j]
        e = vj - vi
        L = np.linalg.norm(e)
        if L < 1e-15: continue
        # outward unit normal relative to triangle orientation:
        # for CCW triangle, interior is on the LEFT of edge (i->j),
        # so outward normal is the RIGHT normal = rotate by -90°
        n_out = np.array([ e[1], -e[0] ]) / L

        mid = 0.5*(vi + vj)
        dfdx, dfdy = grad_f_exact(mid[0], mid[1])
        q = dfdx*n_out[0] + dfdy*n_out[1]           # ∂f/∂n at midpoint
        contrib = 0.5 * L * q                       # midpoint rule, split to i and j
        b[i] += contrib
        b[j] += contrib
    return b

def laplacian_f(vertices, triangles, fvals, grad_f_exact):
    """
    Δ f ≈ - M^{-1} S f + M^{-1} b, with b from boundary flux ∮ φ ∂f/∂n.
    """
    S = cotan_stiffness(vertices, triangles)
    Mdiag = vertex_areas(vertices, triangles)
    b = boundary_flux_load(vertices, triangles, grad_f_exact)
    return ( -S @ fvals + b ) / np.maximum(Mdiag, 1e-16)


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
    lapl_f = laplacian_f(vertices, triangles, f(x, y), grad_f_exact)

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

def main():
    test()

if __name__ == '__main__':
    main()
