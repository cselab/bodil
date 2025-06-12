import numpy as np
import matplotlib.pyplot as plt

# Parameters
n_samples = 10000
b = 1  # banana bend factor
sigma_x = 1.0
sigma_y = 1.0

# Sample from standard normal
x = np.random.normal(0, sigma_x, n_samples)
y = np.random.normal(0, sigma_y, n_samples)

# Apply banana transformation
z = y + b * (x**2 - sigma_x**2)

xmin, xmax = -4, 4
ymin, ymax = -4, 10

xone = np.linspace(xmin, xmax, 64)
yone = np.linspace(ymin, ymax, 128)
X, Y = np.meshgrid(xone, yone)

def compute_logp(X, Y):
    return -X**2 / (2*sigma_x**2) - (Y - b * (X**2 - sigma_x**2))**2 / (2 * sigma_y**2)

logp = compute_logp(X, Y)

# Plot
fig, ax = plt.subplots()
ax.plot(x, z, 'ok', alpha=0.2)
ax.contour(X, Y, logp, levels=20)
ax.set_xlabel('x')
ax.set_ylabel('z = y + b(x² - σ²)')
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
plt.show()
plt.close(fig)


# laplace approximation: max at X, Y = (0, 0)
# covariance:
# d^2logp/dy^2 = -1/sigma_y**2


fig, ax = plt.subplots()
ax.hist(z, range=(ymin, ymax), bins=50, density=True)
ax.plot(yone, np.exp(-yone**2/(2*sigma_y**2)) / np.sqrt(2*np.pi*sigma_y**2), '-r')
plt.show()
