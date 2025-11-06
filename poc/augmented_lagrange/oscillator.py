import numpy as np
import matplotlib.pyplot as plt

# ---- problem setup ----
omega = 1.0
dt = 0.05
N = 120
t = np.linspace(0, N*dt, N+1)

# true trajectory and noisy data
x_true = np.cos(omega * t)
v_true = -omega * np.sin(omega * t)
rng = np.random.default_rng(0)
x_data = x_true + 0.05 * rng.normal(size=x_true.shape)

# ---- variable layout ----
# unknowns z = [x0, v0, x1..xN, v1..vN]  (length 2(N+1))
def unpack(z):
    x = z[:N+1]
    v = z[N+1:]
    return x, v

def pack(x, v):
    return np.concatenate([x, v])

# ---- discrete physics constraints ----
def constraints(x, v):
    c = np.zeros(2*N)
    for k in range(N):
        c[2*k]   = v[k+1] - v[k] + dt*omega**2*x[k]
        c[2*k+1] = x[k+1] - x[k] - dt*v[k+1]
    return c

# ---- Jacobian of constraints ----
def constraint_jacobian(x, v):
    A = np.zeros((2*N, 2*(N+1)))
    for k in range(N):
        # r_v
        A[2*k, N+1+k]   += -1.0        # ∂r_v/∂v_k
        A[2*k, N+1+k+1] +=  1.0        # ∂r_v/∂v_{k+1}
        A[2*k, k]       +=  dt*omega**2# ∂r_v/∂x_k
        # r_x
        A[2*k+1, k]       += -1.0      # ∂r_x/∂x_k
        A[2*k+1, k+1]     +=  1.0      # ∂r_x/∂x_{k+1}
        A[2*k+1, N+1+k+1] += -dt       # ∂r_x/∂v_{k+1}
    return A

# ---- objective: data misfit ----
def objective(x):
    return 0.5*np.sum((x - x_data)**2)

def grad_objective(x, v):
    g = np.zeros(2*(N+1))
    g[:N+1] = x - x_data
    return g

# ---- Newton-KKT solver for constrained problem ----
def kkt_step(z, lam, mu):
    x, v = unpack(z)
    c = constraints(x, v)
    A = constraint_jacobian(x, v)
    gJ = grad_objective(x, v)
    rhs1 = -(gJ + A.T @ (lam + mu*c))
    rhs2 = -c

    # Hessian of J + μAᵀA  (approx SPD)
    H = np.zeros((2*(N+1), 2*(N+1)))
    H[:N+1,:N+1] = np.eye(N+1)
    K11 = H + mu*(A.T@A) + 1e-10*np.eye(2*(N+1))

    # Schur complement solve
    K11_inv_rhs1 = np.linalg.solve(K11, rhs1)
    K11_inv_AT   = np.linalg.solve(K11, A.T)
    S = -(A @ K11_inv_AT)
    rhs_s = rhs2 - A @ K11_inv_rhs1
    dlam = np.linalg.solve(-S + 1e-10*np.eye(2*N), -rhs_s)
    dz = np.linalg.solve(K11, rhs1 - A.T @ dlam)
    return dz, dlam, c

# ---- initialization ----
x_guess = np.zeros(N+1); v_guess = np.zeros(N+1)
# start from random IC guess
x_guess[0], v_guess[0] = 0.7, 0.3
for k in range(N):
    v_guess[k+1] = v_guess[k] - dt*omega**2*x_guess[k]
    x_guess[k+1] = x_guess[k] + dt*v_guess[k]

z = pack(x_guess, v_guess)
lam = np.zeros(2*N)
mu = 0.1

# ---- ALM outer loop ----
prev_cnorm = np.inf
for outer in range(8):
    for inner in range(5):
        dz, dlam, c = kkt_step(z, lam, mu)
        alpha = 1.0
        phi0 = objective(unpack(z)[0]) + lam@c + 0.5*mu*(c@c)
        for _ in range(10):
            z_try = z + alpha*dz
            phi_try = objective(unpack(z_try)[0]) + lam@constraints(*unpack(z_try)) \
                       + 0.5*mu*np.linalg.norm(constraints(*unpack(z_try)))**2
            if phi_try <= phi0 - 1e-6*alpha*(dz@dz): break
            alpha *= 0.5
        z = z_try
        if np.linalg.norm(c) < 1e-6: break
    # update λ
    x_curr, v_curr = unpack(z)
    c_curr = constraints(x_curr, v_curr)
    lam += mu*c_curr
    cnorm = np.linalg.norm(c_curr)
    # adapt μ
    if cnorm < 0.8*prev_cnorm: pass
    elif cnorm > 1.2*prev_cnorm: mu = max(mu*0.5, 1e-4)
    else: mu = min(mu*1.5, 10.0)
    prev_cnorm = cnorm
    print(f"Outer {outer}: ||c||={cnorm:.3e}, mu={mu:.2e}, x0={x_curr[0]:.3f}, v0={v_curr[0]:.3f}")

x_opt, v_opt = unpack(z)

# ---- results ----
print(f"\nRecovered initial conditions: x0={x_opt[0]:.4f}, v0={v_opt[0]:.4f}")

plt.figure()
plt.plot(t, x_true, 'k--', label='True')
plt.plot(t, x_data, 'r.', alpha=0.5, label='Noisy data')
plt.plot(t, x_opt, 'b-', label='Recovered (unknown IC)')
plt.legend(); plt.xlabel('t'); plt.ylabel('x')
plt.title('ODIL (Newton) with Unknown Initial Conditions')
plt.show()
