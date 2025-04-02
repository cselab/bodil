#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

def clamped_uniform_knot_vector(num_ctrl_points, degree, t0=0.0, t1=1.0):
    """Creates a clamped, uniformly spaced knot vector."""
    num_knots = num_ctrl_points + degree + 1
    num_internal = num_knots - 2 * (degree + 1)
    internal_knots = np.linspace(t0, t1, num_internal + 2)[1:-1] if num_internal > 0 else np.array([])
    start = np.full(degree + 1, t0)
    end = np.full(degree + 1, t1)
    return np.concatenate([start, internal_knots, end])


def bspline_basis(t, i, k, knots):
    """Cox–de Boor recursion for B-spline basis function B_{i,k}(t)."""
    if k == 0:
        return np.where((knots[i] <= t) & (t < knots[i + 1]), 1.0, 0.0)
    else:
        denom1 = knots[i + k] - knots[i]
        denom2 = knots[i + k + 1] - knots[i + 1]

        term1 = 0.0
        term2 = 0.0

        if denom1 > 0:
            term1 = (t - knots[i]) / denom1 * bspline_basis(t, i, k - 1, knots)
        if denom2 > 0:
            term2 = (knots[i + k + 1] - t) / denom2 * bspline_basis(t, i + 1, k - 1, knots)

        return term1 + term2


def evaluate_all_basis(t_vals, num_ctrl_points, degree=3, t0=0.0, t1=1.0):
    """Evaluates all B-spline basis functions at given t values."""
    knots = clamped_uniform_knot_vector(num_ctrl_points, degree, t0, t1)
    basis_vals = []
    for i in range(num_ctrl_points):
        B_i = bspline_basis(t_vals, i, degree, knots)
        basis_vals.append(B_i)
    return np.stack(basis_vals, axis=-1), knots


def main():
    num_ctrl_points = 8
    degree = 3
    t_vals = np.linspace(0, 1, 1000)

    basis_matrix, knots = evaluate_all_basis(t_vals, num_ctrl_points, degree)

    plt.figure(figsize=(8, 4))
    for i in range(basis_matrix.shape[1]):
        plt.plot(t_vals, basis_matrix[:, i], label=f'B{i}')
    plt.title("Cubic B-spline Basis Functions (Clamped)")
    plt.xlabel("t")
    plt.ylabel("Basis Value")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
