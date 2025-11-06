import numpy as np
import matplotlib.pyplot as plt

# ---- Problem setup ----
omega = 1.0
dt = 0.05
N = 120
t = np.linspace(0, N*dt, N+1)

# True trajectory
x_true = np.cos(omega * t)
v_true = -omega * np.sin(omega * t)

# Select a few observation indices (sparse data)
ndata = 15
sigma = 0.2
obs_idx = np.random.choice(np.arange(N), size=ndata, replace=False)
rng = np.random.default_rng(0)
x_obs = rng.normal(x_true[obs_idx], sigma)

# ---- Variable layout ----
# z = [x0..xN, v0..vN]
def unpack(z):
    x = z[:N+1]
    v = z[N+1:]
    return x, v

def pack(x, v):
    return np.concatenate([x, v])

# ---- Constraints (symplectic Euler) ----
def constraints(x, v):
    c = np.zeros(2*N)
    for k in range(N):
        c[2*k]   = v[k+1] - v[k] + dt*omega**2*x[k]
        c[2*k+1] = x[k+1] - x[k] - dt*v[k+1]
    return c

def constraints_jacobian(x, v):
    A = np.zeros((2*N, 2*(N+1)))
    for k in range(N):
        # r_v
        A[2*k, N+1+k]   += -1.0
        A[2*k, N+1+k+1] +=  1.0
        A[2*k, k]       +=  dt*omega**2
        # r_x
        A[2*k+1, k]       += -1.0
        A[2*k+1, k+1]     +=  1.0
        A[2*k+1, N+1+k+1] += -dt
    return A

# ---- Sparse-data objective ----
def objective(x):
    r = x[obs_idx] - x_obs
    return 0.5 * np.dot(r, r)

def grad_objective(x, v):
    g = np.zeros(2*(N+1))
    # only gradient entries for observed times
    for i, idx in enumerate(obs_idx):
        g[idx] = x[idx] - x_obs[i]
    return g

def H_objective(x, v):
    H = np.zeros((2*(N+1), 2*(N+1)))
    for i, idx in enumerate(obs_idx):
        H[idx, idx] = 1.0
    return H

def newton_step(z, lam, mu):
    x, v = unpack(z)
    c = constraints(x, v)
    A = constraints_jacobian(x, v)
    gJ = grad_objective(x, v)
    rhs = -(gJ + A.T @ (lam + mu * c))
    H = H_objective(x, v) + mu * A.T @ A
    dz = np.linalg.solve(H, rhs)
    return dz, c

# ---- Initialization ----
x_guess = np.zeros(N+1); v_guess = np.zeros(N+1)

z = pack(x_guess, v_guess)
lam = np.zeros(2 * N)
mu = 0.1



x_curr, v_curr = unpack(z)
c_curr = constraints(x_curr, v_curr)
cnorm = np.linalg.norm(c_curr)
print(f"Outer {0:03d}: ||c||={cnorm:.3e}, mu={mu:.2e}, x0={x_curr[0]:.3f}, v0={v_curr[0]:.3f}, objective={objective(x_curr):.3e}")

# ---- Augmented Lagrangian loop ----
prev_cnorm = np.inf
for outer in range(50):
    for inner in range(5):
        dz, c = newton_step(z, lam, mu)
        z = z + dz
        if np.linalg.norm(c) < 1e-6:
            break

    # λ update and adaptive μ
    x_curr, v_curr = unpack(z)
    c_curr = constraints(x_curr, v_curr)
    lam += mu * c_curr
    cnorm = np.linalg.norm(c_curr)
    if cnorm < 0.8 * prev_cnorm:
        pass
    elif cnorm > 1.2 * prev_cnorm:
        mu = max(mu * 0.5, 1e-4)
    else:
        mu = min(mu * 1.5, 100.0)
    prev_cnorm = cnorm
    print(f"Outer {outer+1:03d}: ||c||={cnorm:.3e}, mu={mu:.2e}, x0={x_curr[0]:.3f}, v0={v_curr[0]:.3f}, objective={objective(x_curr):.3e}")

x_opt, v_opt = unpack(z)
print(f"\nRecovered ICs: x0={x_opt[0]:.4f}, v0={v_opt[0]:.4f}")

# ---- Visualization ----
plt.figure()
plt.plot(t, x_true, 'k--', label='True')
plt.plot(t[obs_idx], x_obs, 'ro', label='Sparse data')
plt.plot(t, x_opt, 'b-', label='Recovered trajectory')
plt.legend(); plt.xlabel('t'); plt.ylabel('x')
plt.title('ODIL (Newton) with sparse data and unknown ICs')
plt.show()
