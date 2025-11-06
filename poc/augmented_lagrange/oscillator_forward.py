import numpy as np
import matplotlib.pyplot as plt

# ------------------------
# Problem setup
# ------------------------
omega = 1.0
dt = 0.05
N = 200          # number of steps
t = np.linspace(0, N*dt, N+1)

# Known initial conditions
x0 = 1.0
v0 = 0.0

# For comparison: analytical solution
x_true = np.cos(omega * t)
v_true = -omega * np.sin(omega * t)

# ------------------------
# Variables: z = [x_1..x_N, v_1..v_N]
# We'll solve r(z) = 0 by Newton: J dz = -r
# ------------------------
def unpack(z):
    x1N = z[:N]
    v1N = z[N:]
    x = np.empty(N+1); v = np.empty(N+1)
    x[0] = x0; v[0] = v0
    x[1:] = x1N
    v[1:] = v1N
    return x, v

def residual_and_jacobian(z, dt, omega):
    """
    r shape: (2N,)
    J shape: (2N, 2N)
    Ordering of residuals:
      for k = 0..N-1:
        r[2k]   = r_v_k   (velocity update residual)
        r[2k+1] = r_x_k   (position update residual)
    Variables are [x_1..x_N, v_1..v_N].
    """
    x, v = unpack(z)
    r = np.zeros(2*N)
    J = np.zeros((2*N, 2*N))

    # helper to map (time index j >= 1) to variable column indices
    def col_x(j): return j-1                  # x_j variable column
    def col_v(j): return N + (j-1)            # v_j variable column

    for k in range(N):
        # Residuals
        r_v = v[k+1] - v[k] + dt * (omega**2) * x[k]
        r_x = x[k+1] - x[k] - dt * v[k+1]
        r[2*k]   = r_v
        r[2*k+1] = r_x

        # Jacobian entries (w.r.t. variables x_1..x_N, v_1..v_N)
        # r_v depends on: v_{k+1}(+1), v_k(-1), x_k(+dt*omega^2)
        # r_x depends on: x_{k+1}(+1), x_k(-1), v_{k+1}(-dt)

        # w.r.t x_{k} if k >= 1
        if k >= 1:
            J[2*k, col_x(k)] += dt * (omega**2)   # ∂r_v/∂x_k
            J[2*k+1, col_x(k)] += -1.0            # ∂r_x/∂x_k
        # w.r.t x_{k+1} (always a variable since k+1 >= 1)
        J[2*k+1, col_x(k+1)] += 1.0               # ∂r_x/∂x_{k+1}

        # w.r.t v_k if k >= 1
        if k >= 1:
            J[2*k, col_v(k)] += -1.0              # ∂r_v/∂v_k
        # w.r.t v_{k+1} (always a variable)
        J[2*k,   col_v(k+1)] +=  1.0              # ∂r_v/∂v_{k+1}
        J[2*k+1, col_v(k+1)] += -dt               # ∂r_x/∂v_{k+1}

    return r, J

# ------------------------
# Newton solve
# ------------------------
# Initialize with something simple (e.g., propagate with explicit Euler to get a start)
x_guess = np.zeros(N+1); v_guess = np.zeros(N+1)
x_guess[0], v_guess[0] = x0, v0
for k in range(N):
    # crude forward Euler to get a starting trajectory
    v_guess[k+1] = v_guess[k] - dt * (omega + 0.1)**2 * x_guess[k]
    x_guess[k+1] = x_guess[k] + dt * v_guess[k]

z = np.concatenate([x_guess[1:], v_guess[1:]])

for it in range(5):  # should converge in 1 step for this linear residual
    r, J = residual_and_jacobian(z, dt, omega)
    rn = np.linalg.norm(r)
    print(f"Newton iter {it}: ||r|| = {rn:.3e}")
    if rn < 1e-12:
        break
    # Solve J dz = -r (residual is linear -> this drives r to ~0 in one step)
    dz = np.linalg.solve(J, -r)
    z = z + dz

# Reconstruct trajectory
x_opt, v_opt = unpack(z)

# ------------------------
# Diagnostics & Plot
# ------------------------
print("\nFinal residual norm:", np.linalg.norm(residual_and_jacobian(z, dt, omega)[0]))
print("IC recovered:", x_opt[0], v_opt[0], "(should equal x0,v0)")

plt.figure()
plt.plot(t, x_true, 'k--', label='Analytic x(t)')
plt.plot(t, x_opt,  'b-',  label='ODIL-Newton (x)')
plt.legend(); plt.xlabel('t'); plt.ylabel('x')
plt.title('Harmonic Oscillator: ODIL via Newton on Discrete Residuals')
plt.show()
