import numpy as np

def f(x):
    """Objective function: sphere."""
    return x[0]**2 + x[1]**2

def grad_f(x):
    """Gradient of f."""
    return np.array([2*x[0], 2*x[1]])

def hess_f(x):
    """Hessian of f."""
    return np.eye(2) * 2

def c(x):
    """Constraint: x + y - 1 = 0."""
    return np.array([x[0] + x[1] - 1])

def grad_c(x):
    """Jacobian (gradient) of the constraint."""
    return np.array([[1.0, 1.0]])

def augmented_lagrangian(x, lam, mu):
    """Compute L_aug = f + λᵀc + (μ/2)||c||²."""
    return f(x) + lam @ c(x) + 0.5 * mu * np.sum(c(x)**2)

def grad_L(x, lam, mu):
    """Gradient of augmented Lagrangian."""
    return grad_f(x) + grad_c(x).T @ (lam + mu * c(x))

def hess_L(x, mu):
    """Hessian of augmented Lagrangian."""
    return hess_f(x) + mu * grad_c(x).T @ grad_c(x)

# Initialization
x = np.array([2.0, -1.0])   # initial guess
lam = np.array([0.0])       # Lagrange multiplier
mu = 1.0                    # penalty parameter

for outer_iter in range(10):
    # Inner Newton iteration to minimize L_aug for fixed λ, μ
    for inner_iter in range(10):
        g = grad_L(x, lam, mu)
        H = hess_L(x, mu)
        step = np.linalg.solve(H, -g)
        x = x + step
        if np.linalg.norm(step) < 1e-8:
            break

    # Update λ
    lam = lam + mu * c(x)

    # Increase μ (penalty)
    mu *= 2.0

    print(f"Iter {outer_iter}: x = {x}, λ = {lam}, c(x) = {c(x)}, f(x) = {f(x):.6f}")

print("\nFinal result:")
print(f"x* = {x}")
print(f"Constraint value = {c(x)}")
print(f"f(x*) = {f(x):.6f}")
